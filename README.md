# DeepSort-check

# download python IDE and make env
1. download anaconda
2. check all
3. execute Anaconda powershell prompt
4.  typing this code to install & execute virtual enviroment
conda create -n deepsort python
conda activate deepsort
5. execute jupyter Notebook
6. select Kernel ( Python [conda env:your environment])


# use deepSORT
1. cloning recurse submodules
   git clone --recurse-submodules https://github.com/ultralytics/yolov5.git

2. install YOLOv5 & requirement
   git clone https://github.com/ultralytics/yolov5.git
   cd yolov5 
   pip install -r requirements

3. install python packages
   pip insttall numpy opencv-python onnxruntime
   pip install deep_sort_realtime




#YOLOv5 model import & test.
#use \\ not \
#(1). use official dataset

import torch

model=torch.hub.load('ultralytics/yolov5', 'yolov5s',pretrained=True)

img_path='C:\\Users\\KOSTA\\Downloads\\cat.jpg'

results=model(img_path)

results.print()
results.show()


#(2). use my dataset
from roboflow import Roboflow
import torch

#1. download Roboflow Dataset
rf = Roboflow(api_key="QuSmgEUYSEhArO48H4LV")
project = rf.workspace("search-buyer").project("peopledetection-qyok3-dlf77")
version = project.version(5)
dataset = version.download("yolov5")

#2. load YOLOc5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

#3. Insurance test image.
img_path= "C:\\Users\\KOSTA\\Downloads\\pilgi.jpg"
results=model(img_path)

results.print()
results.show()
