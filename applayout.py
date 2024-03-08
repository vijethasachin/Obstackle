from kivy.core.window import Window
from kivy.lang import Builder
from kivy.properties import StringProperty, ObjectProperty
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.utils import platform
from classifyobject import ClassifyObject

class AppLayout(FloatLayout):
    detect = ObjectProperty()
        

Builder.load_string("""
<AppLayout>:
    detect: self.ids.preview
    ClassifyObject:
        letterbox_color: 'black'
        id:preview
    
""")

            
