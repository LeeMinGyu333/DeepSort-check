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


#(2). use my dataset( in windows , pi에서 할 시 경로만 바꾸면 잘될거임.)
import torch
import cv2
import numpy as np

# model load
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/KOSTA/best.pt', force_reload=True, device='cpu')

# image load and preprocessing
img = cv2.imread('C:/Users/KOSTA/Downloads/mingyu2.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# into the model (support Auto preprocessing)
results = model(img)

# output answer.
results.print()
results.show()




#(2-1). use my dataset( in Ubuntu(Raspberry pi))
import torch

model=torch.hub.load('home/pi/yolov5','custom',path='home/pi/best.pt',source='local')


img_path='?/cat.jpg'

results=model(img_path)

results.print()
results.show()
results.show()




#Tracking in PC(windows) in DeepSORT?
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import os
import random
import cv2
import torch

import pathlib
import platform


if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

from deep_sort_realtime.deepsort_tracker import DeepSort  # DeepSORT 설치 필요

# 웹캠 열기 (기본 카메라: 0)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

if not cap.isOpened():
    print("웹캠 열기에 실패했습니다.")
    exit()

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', device='cpu', force_reload=True)
print("YOLOv5 모델 로드 완료")

# DeepSORT 초기화
tracker = DeepSort(max_age=30)

# 색상 팔레트 생성
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(50)]

detection_threshold = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임 읽기에 실패했습니다.")
        break

    # YOLOv5는 BGR 이미지를 받음
    results = model(frame)

    detections = []
    for result in results.xyxy[0].tolist():
        x1, y1, x2, y2, score, class_id = result
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        if score > detection_threshold:
            detections.append(([x1, y1, x2 - x1, y2 - y1], score, 'person'))

    # tracker.update expects: list of [tlwh, confidence, class]
    tracks = tracker.update_tracks(detections, frame=frame)

    # 트래킹 결과 시각화
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, r, b = track.to_ltrb()
        color = colors[int(track_id) % len(colors)]
        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), color, 3)
        cv2.putText(frame, f"ID: {track_id}", (int(l), int(t) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 결과 출력
    cv2.imshow('Real-time Tracking', frame)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



