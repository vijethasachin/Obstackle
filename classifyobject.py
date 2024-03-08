import time
from kivy.clock import mainthread
from kivy.graphics import Color, Rectangle
from kivy.core.audio import SoundLoader
import numpy as np
from camera4kivy import Preview

from kivy.graphics.texture import Texture
from plyer import tts
import PIL
from tflite_runtime.interpreter import Interpreter
import cv2
from PIL import Image


class DisparityEstimator:

  def __init__(self):
    self.road = None
    self.depthinterpreter = Interpreter(model_path="models/midas.tflite")
    self.depthinterpreter.allocate_tensors()
    self.input_details = self.depthinterpreter.get_input_details()
    self.output_details = self.depthinterpreter.get_output_details()
    self.op_index = 0
    self.disp=None

  
  #Performs road profile elimination on the depth data and returns the rest
  def getProcessedDepthImage(self, img):
        img = self.preprocess(img)
        self.disp = self.getDisparity(img)
        disp_thresh = 0.45*np.max(self.disp)
        v_disp = self.getVDisp(self.disp)
        canny_edges = cv2.Canny(v_disp.astype(np.uint8), 30, np.max(v_disp))
        canny_L = self.getLCanny(canny_edges)
        slope, c = self.getHoughLine(canny_L)
        self.road = self.extractRoadProfile(self.disp, slope, c)
        self.road[self.road<disp_thresh]=0
        return self.road


  def getVDisp(self, disp):
    disp = np.array(disp, dtype=np.uint8)
    max_disp = np.max(disp) + 1
    indices = disp >= 0
    non_negative_disp = disp[indices] - 1
    rows = np.where(indices)[0]
    v_disp = np.zeros((disp.shape[0], max_disp), dtype=float)
    np.add.at(v_disp, (rows, non_negative_disp), 1)
    return v_disp


  def getLCanny(self, canny_edges):
        canny_L = canny_edges.copy()
        for i in range(canny_edges.shape[0]):
            for j in (range(canny_edges.shape[1])):
                if (canny_edges[i][j] > 0):
                    canny_L[i][j] = canny_edges[i][j]
                    canny_L[i][j + 1:] = 0
                    break
        return canny_L

  def pointOnLine(self, x, y, m, c, thres):
      distances = np.abs(y - (m * x + c))
      return (distances < thres).astype(int)

  def getHoughLine(self, edges):
        lines = cv2.HoughLines(edges, 4, np.pi / 180, 30)
        if type(lines) == type(None):
            return -np.inf, -np.inf
        highest_slope = -np.inf
        x1_low, y1_low, x2_low, y2_low = 1, 1, 0, 0
        highest_points_on_line = -np.inf
        num=0
        for line in lines:
            num=num+1
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            if (x2 == x1):
                continue
            else:
                slope = (y2 - y1) / (x2 - x1)

            points_on_line = 0

            for u_ in range(max(0, min(x1, x2)), min(edges.shape[1], max(x1, x2))):
                try:
                    v_ = int((rho - u_ * np.cos(theta)) / np.sin(theta))
                    if v_ >= 0 and v_ < edges.shape[0] and edges[v_, u_] > 0:
                        points_on_line += 1
                except:
                    print("u:", u_, " v:", v_)

            if points_on_line>=highest_points_on_line and slope>=0:

                if y2 < y1 and x2 < x1:
                    #print("Line length: ", points_on_line)
                    continue

                highest_slope = slope
                highest_points_on_line = points_on_line

                x1_low, y1_low = x1, y1
                x2_low, y2_low = x2, y2

        slope = (y2_low - y1_low) / (x2_low - x1_low)
        c = y1_low - slope * x1_low
        return slope, c
  
  #additional road profile extraction to deal with noisy segmentation data 
  #generated from real world images
  def extractRoadProfile(self, disp, slope, c):
    disp = np.array(disp, dtype=np.uint8)
    u_values = np.arange(disp.shape[0])
    v_values = np.arange(disp.shape[1])
    u_grid, v_grid = np.meshgrid(u_values, v_values, indexing='ij')
    # Calculate modified disp array by subtracting 1 from each element
    modified_disp = disp - 1
    # Compute the condition mask using self.pointOnLine function
    condition_mask = self.pointOnLine(modified_disp, u_grid, slope, c, 30)
    # Update 'road' with zeros where the condition is True
    road = np.where(condition_mask, 0, disp)
    return road

  
  def preprocess(self, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    img_input = cv2.resize(img, (256,256), interpolation=cv2.INTER_CUBIC)
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    img_input = (img_input - mean) / std
    return img_input
  
  # Extract disparity image from the captured image
  def getDisparity(self, img):
        input_data = np.array([cv2.resize(img, (256, 256))]).astype("float32")
        self.depthinterpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.depthinterpreter.invoke()
        norm = np.zeros((256,256))
        output_data = self.depthinterpreter.get_tensor(self.output_details[self.op_index]['index'])
        disp_img = cv2.normalize(output_data[0], norm, 0, 255, norm_type=cv2.NORM_MINMAX)
        return disp_img



class SemanticSegmentor:
  def __init__(self):
    self.image=None
    self.colormap = self.create_ade20k_label_colormap()
    self.seg_interpreter = Interpreter(model_path='models/topformer.tflite')
    self.seg_interpreter.allocate_tensors()  # Get input and output tensors.
    self.input_details = self.seg_interpreter.get_input_details()
    self.output_details = self.seg_interpreter.get_output_details()

  def getSegments(self, img):
    self.image = img
    return self.getTopFrSegments(self.image)

  def create_ade20k_label_colormap(self):
    """Creates a label colormap used in ADE20K segmentation benchmark.
    Returns:
      A colormap for visualizing segmentation results.
    """
    return np.asarray([
        [0, 0, 0],      [120, 120, 120],      [180, 120, 120],      [6, 230, 230],         [80, 50, 50],      [4, 200, 3],      [120, 120, 80],      [140, 140, 140],
        [204, 5, 255],      [230, 230, 230],      [4, 250, 7],      [224, 5, 255],         [235, 255, 7],      [150, 5, 61],      [120, 120, 70],      [8, 255, 51],
        [255, 6, 82],      [143, 255, 140],      [204, 255, 4],      [255, 51, 7],         [204, 70, 3],      [0, 102, 200],      [61, 230, 250],      [255, 6, 51],
        [11, 102, 255],      [255, 7, 71],      [255, 9, 224],      [9, 7, 230],         [220, 220, 220],      [255, 9, 92],      [112, 9, 255],      [8, 255, 214],
        [7, 255, 224],      [255, 184, 6],      [10, 255, 71],      [255, 41, 10],      [7, 255, 255],         [224, 255, 8],      [102, 8, 255],      [255, 61, 6],      [255, 194, 7],
        [255, 122, 8],      [0, 255, 20],      [255, 8, 41],      [255, 5, 153],         [6, 51, 255],      [235, 12, 255],      [160, 150, 20],      [0, 163, 255],
        [140, 140, 140],      [250, 10, 15],      [20, 255, 0],      [31, 255, 0],         [255, 31, 0],      [255, 224, 0],      [153, 255, 0],      [0, 0, 255],      [255, 71, 0],
        [0, 235, 255],      [0, 173, 255],      [31, 0, 255],      [11, 200, 200],      [255, 82, 0],         [0, 255, 245],      [0, 61, 255],      [0, 255, 112],      [0, 255, 133],
        [255, 0, 0],      [255, 163, 0],      [255, 102, 0],      [194, 255, 0],      [0, 143, 255],         [51, 255, 0],      [0, 82, 255],      [0, 255, 41],      [0, 255, 173],
        [10, 0, 255],      [173, 255, 0],      [0, 255, 153],      [255, 92, 0],         [255, 0, 255],      [255, 0, 245],      [255, 0, 102],      [255, 173, 0],
        [255, 0, 20],      [255, 184, 184],      [0, 31, 255],      [0, 255, 61],         [0, 71, 255],      [255, 0, 204],      [0, 255, 194],      [0, 255, 82],
        [0, 10, 255],      [0, 112, 255],      [51, 0, 255],      [0, 194, 255],        [0, 122, 255],      [0, 255, 163],      [255, 153, 0],      [0, 255, 10],      [255, 112, 0],
        [143, 255, 0],      [82, 0, 255],      [163, 255, 0],      [255, 235, 0],      [8, 184, 170],         [133, 0, 255],      [0, 255, 92],      [184, 0, 255],      [255, 0, 31],
        [0, 184, 255],      [0, 214, 255],      [255, 0, 112],      [92, 255, 0],      [0, 224, 255],         [112, 224, 255],      [70, 184, 160],      [163, 0, 255],      [153, 0, 255],
        [71, 255, 0],      [255, 0, 163],      [255, 204, 0],      [255, 0, 143],         [0, 255, 235],      [133, 255, 0],      [255, 0, 235],      [245, 0, 255],
        [255, 0, 122],      [255, 245, 0],      [10, 190, 212],      [214, 255, 0],         [0, 204, 255],      [20, 0, 255],     [255, 255, 0],      [0, 153, 255],
        [0, 41, 255],      [0, 255, 204],      [41, 0, 255],      [41, 255, 0],         [173, 0, 255],      [0, 245, 255],      [71, 0, 255],      [122, 0, 255],
        [0, 255, 184],      [0, 92, 255],      [184, 255, 0],      [0, 133, 255],         [255, 214, 0],      [25, 194, 194],      [102, 255, 0],      [92, 0, 255],
    ])


  def label_to_color_image(self, label):
    if label.ndim != 2:
      raise ValueError('Expect 2-D input label')

    if np.max(label) >= len(self.colormap):
      label= np.where(label >=len(self.colormap), 0, label)

    return self.colormap[label]


  def input_transform(self, image):
      mean=[123.675, 116.28, 103.53]
      std=[58.395, 57.12, 57.375]
      image = image.astype(np.float32)[:, :, ::-1]
      image -= mean
      image /= std
      return np.array(image)


  def generateTFSegments( self, pil_image, interpreter):
      pil_image = Image.fromarray(pil_image)
      input_size = self.input_details[0]['shape'][2], self.input_details[0]['shape'][1]
      new_img = pil_image.resize((input_size[0], input_size[1]))
      np_new_img = np.array(new_img)
      np_new_img=self.input_transform(np_new_img)
      np_new_img = np.expand_dims(np_new_img, axis=0)
      interpreter.set_tensor(self.input_details[0]['index'], np_new_img)
      interpreter.invoke()
      raw_prediction = interpreter.tensor(interpreter.get_output_details()[0]['index'])()
      seg_map = np.squeeze(raw_prediction).astype(np.int8)
      seg_map = np.asarray(Image.fromarray(seg_map).resize((256,256),resample=0))
      return seg_map


  def getTopFrSegments(self, img):
      img = cv2.medianBlur(img, ksize=5)
      seg_map = self.generateTFSegments(img, self.seg_interpreter)
      seg_map = np.array(seg_map)
      seg_map = seg_map+1
      combineArr = [4, 7,12,14,53,29,30,92, 95]
      for item in combineArr:
          seg_map[seg_map == item] = 4
      return seg_map

  #Generate the masks for the 6 regions defined in the image
  def getMasks(self, mask):
        w = mask.shape[1]
        h = mask.shape[0]
        masks = []

        mask1 = mask.copy()
        mask1[int(h * 0.1):int(h * 0.4), int(w * 0):int(w * 0.2)] = 255
        masks.append(mask1)

        mask1 = mask.copy()
        mask1[int(h * 0.1):int(h * 0.4), int(w * 0.2):int(w * 0.8)] = 255
        masks.append(mask1)

        mask1 = mask.copy()
        mask1[int(h * 0.1):int(h * 0.4), int(w * 0.8):int(w * 1)] = 255
        masks.append(mask1)

        mask1 = mask.copy()
        mask1[int(h * 0.6):int(h), int(w * 0):int(w * 0.2)] = 255
        masks.append(mask1)

        mask1 = mask.copy()
        mask1[int(h * 0.6):int(h), int(w * 0.2):int(w * 0.8)] = 255
        masks.append(mask1)

        mask1 = mask.copy()
        mask1[int(h * 0.6):int(h), int(w * 0.8):int(w * 1)] = 255
        masks.append(mask1)
        return masks

  # Determine the obstacle with the maximum strength in the middle segment
  def getProminentObstacle(self, seg_map):
        # construct obstacle mask
        mask = np.zeros((seg_map.shape))
        w = mask.shape[1]
        h = mask.shape[0]

        # centre mask
        mask[int(h * 0.1):int(h), int(w * 0.2):int(w * 0.8)] = 1
        mask = np.array(mask).astype(np.uint8)
        centre_mask = cv2.bitwise_and(seg_map, seg_map, mask=mask)
        maxStrength = -1
        strongLabel = -1
        for label in np.unique(centre_mask):
            if (label == 0):
                continue
            label_mask = (centre_mask == label).astype("uint8") * 255  # np.where(seg_map == label,255,0)
            label_strength = np.array(label_mask).flatten().tolist().count(255)

            if (label_strength > maxStrength):
                maxStrength = label_strength
                strongLabel = label
        return strongLabel




class ClassifyObject(Preview):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # ====================
        self.seg_map = None
        self.disp_img = None
        self.disp_thresh = None
        self.prominentObstacle = None
        self.origImg = None
        self.analyzed_texture = None
        self.frame = None
        self.classified = None
        self.frameCount = 0
        self.actionPool=[]
        self.prevDrn = -1
        self.dispEstimator = DisparityEstimator()
        self.semSegmentor = SemanticSegmentor()
        self.sound = SoundLoader.load('audio/beep.wav')
        self.labels_list = ['others','wall', 'building','sky','floor','tree','ceiling',
                            'road', 'bed','windowpane','grass','cabinet','sidewalk','person','earth','door',
                            'table','mountain','plant','curtain','chair','car','water','painting','sofa',
                            'shelf','house','sea','mirror','rug','field','armchair','seat','fence','desk''rock',
                            'wardrobe','lamp','bathtub','railing','cushion','base','box','column','signboard','chest',
                            'counter','sand','sink','skyscraper','fireplace','refrigerator','grandstand','path',
                            'stairs','runway','case','pool','pillow','screen','stairway','river','bridge','bookcase',
                            'blind','coffee','toilet','flower','book','hill','bench','countertop','stove','palm',
                            'kitchen','computer','swivel','boat','bar','arcade','hovel','bus','towel','light','truck',
                            'tower','chandelier','awning','streetlight','booth','television','airplane','dirt','apparel',
                            'pole','land','bannister','escalator','ottoman','bottle','buffet','poster','stage','van','ship',
                            'fountain','conveyer','canopy','washer','plaything','swimming','stool','barrel','basket','waterfall','tent',
                            'bag','minibike','cradle','oven','ball','food','step','tank','trade','microwave','pot','animal','bicycle','lake',
                            'dishwasher','screen','blanket','sculpture','hood','sconce','vase','traffic','tray','ashcan','fan','pier',
                            'crt','plate','monitor','bulletin','shower','radiator','glass','clock','flag']


    def discardFartherObstacles(self, seg_map_op, disp_img):
        disp_thresh = self.disp_thresh
        componentMask = np.zeros(seg_map_op.shape, dtype='uint8')
        disp_img = np.array(disp_img).astype("uint8")

        for label in np.unique(seg_map_op):
            label_mask = (seg_map_op == label).astype("uint8") * 255  # np.where(seg_map == label,255,0)

            # 1. Divide seg image into indep component irrespective of label
            threshold = cv2.threshold(label_mask, 0, 255, cv2.THRESH_TOZERO | cv2.THRESH_OTSU)[1]
            analysis = cv2.connectedComponentsWithStats(threshold, 4, cv2.CV_32S)
            (totalLabels, label_ids, values, centroid) = analysis

            # 2. discard smaller and distant segments
            for i in range(1, totalLabels):
                # prepare a mask of this segment t check disparity value

                temp_mask = (label_ids == i).astype("uint8") * 255  # np.where(seg_map == label,255,0)
                temp_disp = cv2.bitwise_and(temp_mask,disp_img, mask = None)

                max_disp = np.max(temp_disp)#np.mean([temp_disp[temp_disp > 0]])
                # prepare a component mask to get the final obstacle and road segments
                if (max_disp > disp_thresh):
                    componentMask = componentMask + temp_mask
        seg_map_op = cv2.bitwise_and(seg_map_op.astype("uint8"), componentMask, mask = disp_img)
        return seg_map_op



    def get_processed_depth_image(self, img):
        self.disp_img = self.dispEstimator.getProcessedDepthImage(img)

    def get_segments(self, img):
        #start = time.time()()
        self.seg_map = self.semSegmentor.getSegments(img)
        #print("Seg time: ", time.time()-start, " seconds")

    def detect(self, input_image: np.ndarray):
        #1 Read image
        self.origImg = input_image.copy()
        self.origImg = cv2.resize(self.origImg, (256, 256))

        self.get_processed_depth_image(self.origImg)
        self.get_segments(self.origImg)
        self.disp_thresh = np.max(self.disp_img)*0.6

        #4 Discard farther segments
        seg_map = self.discardFartherObstacles(self.seg_map, self.disp_img)
        # get display image
        self.buildFinalImage(seg_map)

        #6 Get Prominent Obstacle
        self.prominentObstacle = self.semSegmentor.getProminentObstacle(seg_map)

        if self.prominentObstacle not in [-1,4]:
            #7 compute OStatus
            ostatus = self.computeOstatus(seg_map)
            # #print("Status: ", ostatus)

            #9 Invoke Fuzzzifier
            direction = self.fuzzify(ostatus)

            sectors = ['left', 'ahead', 'right']

            obstacle = self.labels_list[self.prominentObstacle].split(';')[0]
            msg = obstacle + " ahead. Move "+ sectors[direction]

        else:
            direction = 1
            msg = "Move ahead."

        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (10, 20)
        fontScale = 0.25
        color = (255, 255, 255)
        thickness = 1
        self.frame = cv2.putText(self.frame, msg, org, font, fontScale, color, thickness,
                                     cv2.LINE_AA)

        if direction != self.prevDrn:
           tts.speak(msg)

        else:
            if direction!=1:
                tts.speak(msg)
            else:
                self.sound.play()
        self.prevDrn = direction
        return msg

    def trapmf(self, x, params):
        a, b, c, d = params
        result = np.zeros_like(x)

        mask_left = (x >= a) & (x < b)
        mask_mid = (x >= b) & (x <= c)
        mask_right = (x > c) & (x <= d)

        result[mask_left] = (x[mask_left] - a) / (b - a)
        result[mask_mid] = 1
        result[mask_right] = (d - x[mask_right]) / (d - c)
        return result

    def interp_membership(self, x, mf, val):
        return np.interp(val, x, mf)

    def fuzzify(self, status):
        topleft = np.arange(0, 1, 0.1)
        topmid = np.arange(0, 1, 0.1)
        topright = np.arange(0, 1, 0.1)
        botleft = np.arange(0, 1, 0.1)
        botmid = np.arange(0, 1, 0.1)
        botright = np.arange(0, 1, 0.1)
        actions = np.arange(0, 3, 1)

        # Define membership functions
        tl_nobs = self.trapmf(topleft, [0, 0, 0.5, 0.7])
        tl_obs = self.trapmf(topleft, [0.6, 0.8, 1, 1])
        tm_nobs = self.trapmf(topmid, [0, 0, 0.5, 0.7])
        tm_obs = self.trapmf(topmid, [0.6, 0.8, 1, 1])
        tr_nobs = self.trapmf(topright, [0, 0, 0.5, 0.7])
        tr_obs = self.trapmf(topright, [0.6, 0.8, 1, 1])
        bl_nobs = self.trapmf(botleft, [0, 0, 0.2, 0.3])
        bl_obs = self.trapmf(botleft, [0.2, 0.4, 1, 1])
        bm_nobs = self.trapmf(botmid, [0, 0, 0.1, 0.3])
        bm_obs = self.trapmf(botmid, [0.1, 0.4, 1, 1])
        br_nobs = self.trapmf(botright, [0, 0, 0.2, 0.3])
        br_obs = self.trapmf(botright, [0.2, 0.4, 1, 1])

        moveleft = self.trapmf(actions, [0, 0, 0.4, 0.66])
        moveahead = self.trapmf(actions, [0.55, 0.7, 1.2, 1.33])
        moveright = self.trapmf(actions, [1.23, 1.4, 2, 2])

        # Calculate membership degrees
        tl_nobs_mem = self.interp_membership(topleft, tl_nobs, status[0][0])
        tl_obs_mem = self.interp_membership(topleft, tl_obs, status[0][0])*0.4
        tm_nobs_mem = self.interp_membership(topmid, tm_nobs, status[0][1])
        tm_obs_mem = self.interp_membership(topmid, tm_obs, status[0][1])*0.4
        tr_nobs_mem = self.interp_membership(topright, tr_nobs, status[0][2])
        tr_obs_mem = self.interp_membership(topright, tr_obs, status[0][2])*0.4
        bl_nobs_mem = self.interp_membership(botleft, bl_nobs, status[1][0])
        bl_obs_mem = self.interp_membership(botleft, bl_obs, status[1][0])
        bm_nobs_mem = self.interp_membership(botmid, bm_nobs, status[1][1])
        bm_obs_mem = self.interp_membership(botmid, bm_obs, status[1][1])
        br_nobs_mem = self.interp_membership(botright, br_nobs, status[1][2])
        br_obs_mem = self.interp_membership(botright, br_obs, status[1][2])

        # Define fuzzy rules
        # Define fuzzy rules
        rule1 = np.fmax(bm_obs_mem, tm_obs_mem)
        rule2 = np.fmax(np.fmax(bl_obs_mem, tl_obs_mem), np.fmin(bm_nobs_mem, tm_nobs_mem))
        rule3 = np.fmax(np.fmax(tr_obs_mem, br_obs_mem), np.fmin(bm_nobs_mem, tm_nobs_mem))

        ml_activation = np.fmin(rule2, moveleft)
        ma_activation = np.fmin(rule1, moveahead)
        mr_activation = np.fmin(rule3, moveright)

        aggregated = np.fmax(ml_activation, np.fmax(ma_activation, mr_activation))
        return np.argmin(aggregated)


    def computeOstatus(self, seg_map):
        mask = np.zeros(seg_map.shape)
        w = mask.shape[1]
        h = mask.shape[0]
        masks = self.semSegmentor.getMasks(mask)
        status = [0, 0, 0, 0, 0, 0]
        i_ = 0

        sizes = [w * 0.2 * h * 0.3, w * 0.6 * h * 0.3, w * 0.2 * h * 0.3, w * 0.2 * h * 0.4, w * 0.6 * h * 0.4,
                 w * 0.2 * h * 0.4]
        sizes = np.array(sizes)
        # #print(type(sizes))
        for mask in masks:
            mask = np.array(mask).astype(np.uint8)
            temp = cv2.bitwise_and(seg_map, seg_map, mask=mask)
            status[i_] = len(temp[temp > 0]) / sizes[i_]
            i_ += 1
        status = np.reshape(status, (2, 3))
        return status


    def buildFinalImage(self, seg_map):
        seg_image = self.semSegmentor.label_to_color_image(seg_map).astype(np.uint8)
        w = seg_image.shape[1]
        h = seg_image.shape[0]

        seg_image = self.origImg*0.6 + seg_image*0.4

        temp_image = cv2.rectangle(seg_image, (int(w * 0), int(h * 0.1)), (int(w * 0.2), int(h * 0.3)), (255, 255, 255), 1)
        temp_image = cv2.rectangle(temp_image, (int(w * 0.2), int(h * 0.1)), (int(w * 0.8), int(h * 0.3)), (255, 255, 255), 1)
        temp_image = cv2.rectangle(temp_image, (int(w * 0.8), int(h * 0.1)), (int(w * 0.999), int(h * 0.3)), (255, 255, 255), 1)
        temp_image = cv2.rectangle(temp_image, (int(w * 0), int(h * 0.6)), (int(w * 0.2), int(h * 0.999)), (255, 255, 255),1)
        temp_image = cv2.rectangle(temp_image, (int(w * 0.2), int(h * 0.6)), (int(w * 0.8), int(h * 0.999)), (255, 255, 255), 1)
        temp_image = cv2.rectangle(temp_image, (int(w * 0.8), int(h * 0.6)), (int(w * 0.999), int(h * 0.999)), (255, 255, 255), 1)
        self.frame = temp_image





    def analyze_pixels_callback(self, pixels, image_size, image_pos,
                                image_scale, mirror):
        # Convert pixels to numpy rgb
        # pil_image = PIL.Image.frombytes(mode='RGB', size=image_size,data=pixels)

        rgba = np.fromstring(pixels, np.uint8).reshape(image_size[1], image_size[0], 4)
        rgb = rgba[:, :, :3]
        result_text = self.detect(rgb)
        self.frame = np.asarray(PIL.Image.fromarray(self.frame.astype(np.uint8)).resize(image_size, resample=0))
        pixels = self.frame.tostring()
        self.make_thread_safe(pixels, image_size)

    @mainthread
    def make_thread_safe(self, pixels, size):
        if not self.analyzed_texture or \
                self.analyzed_texture.size[0] != size[0] or \
                self.analyzed_texture.size[1] != size[1]:
            self.analyzed_texture = Texture.create(size=size, colorfmt='rgb')
            self.analyzed_texture.flip_vertical()
        self.analyzed_texture.blit_buffer(pixels, colorfmt='rgb')

    ################################
    # Canvas Update  - on UI Thread
    ################################

    def canvas_instructions_callback(self, texture, tex_size, tex_pos):
        # Add the analysis annotations

        if self.analyzed_texture:
            Color(1, 1, 1, 1)
            Rectangle(texture=self.analyzed_texture,
                      size=tex_size, pos=tex_pos)

