# Obs-tackle
Android App to identify Obstacles in real time from an input image

## Introduction
This repository provides the implementation of the Obstacle dection and identification app introduced in the in the following paper:
<a href="https://link.springer.com/article/10.1007/s00138-023-01499-8">Obs-tackle: an obstacle detection system to assist navigation of visually impaired using smartphones<a>

## Overview
The proposed method combines depth data and semantic segmentation data to selectively extract obstacles within a given distance threshold. The process of extraction is shown in the illustration below.

<picture>
  <img width = "700px" alt="Extraction of obstacle data from depth and semantic segmentation data" 
    src="https://media.springernature.com/full/springer-static/image/art%3A10.1007%2Fs00138-023-01499-8/MediaObjects/138_2023_1499_Fig4_HTML.png?as=webp">
</picture>

Once the obstacles are identified, the image frame is divided into 6 regions and a fuzzy logic algorithm is applied to determine the most suitable path to proceed with for navigation.

<picture>
<img width = "700px" src="https://media.springernature.com/full/springer-static/image/art%3A10.1007%2Fs00138-023-01499-8/MediaObjects/138_2023_1499_Fig9_HTML.png?as=webp"/>
</picture>

The solution is implemented as a smartphone app using Python, kivy, and Buildozer framework.

## Project dependencies
- Python3
- <a href="https://pypi.org/project/Kivy/">kivy</a>
- <a href="https://pypi.org/project/camera4kivy/">camera4kivy</a>
- <a href="https://pypi.org/project/gestures4kivy/">gestures4kivy</a>
- <a href="https://pypi.org/project/tflite-runtime/">tflite-runtime</a>
- <a href="https://pypi.org/project/buildozer/">buildozer</a>

## Other Assets
- tflite versions of suitable depth estimation model</a> and semantic segmentation models. Our project uses <a href="https://github.com/isl-org/MiDaS">MiDaS</a> model for depth estimation and <a href="https://github.com/hustvl/TopFormer">TopFormer</a> model for semantic segmentation.
- Details on conversion of models to tflite can be found <a href="https://www.tensorflow.org/lite/models/convert/convert_models">here</a>. Converted models should be placed inside _models_ directory.
## Preparing your device
- Enable developer settings (search internet for "<device> enable developer settings").
- Enable USB debugging in the developer settings.
- Connect the device via USB to your build host.

## Building the APK
- Run the following command on the terminal

`buildozer android debug deploy run`

Eventually, an APK will be constructed. buildozer proceeds to deploy the APK onto the linked Android device using adb and initiates it.

Additionally, the APK is saved within your project under the bin directory and can be reinstalled on the Android device using `adb install <APK file>`.

## Acknowledgement
The starter code for App development using kivy was borrowed from <a href="https://github.com/Android-for-Python/c4k_tflite_example">c4k_tflite_example</a>.

