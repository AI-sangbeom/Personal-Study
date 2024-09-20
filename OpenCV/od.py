import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

def drawing_bbox(img, result, name_dict):
    boxes = result[0].boxes
    for i, bbox in enumerate(boxes.xyxy):
        x,y,w,h = [int(coord) for coord in bbox[:4]]
        img = cv.rectangle(img, (int(x), int(y)), (int(w), int(h)), (255,0,0), 2)
        label = f'{name_dict[boxes.cls[i].int().item()]} {boxes.conf[i]:.2f}'
        cv.putText(img, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return img

model = YOLO('yolov8n.yaml')
model = YOLO('yolov8n.pt')
# model.train(data='coco128.yaml', epochs=100)

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

try:
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    print(f'width: {width}, height: {height}') # width, height)

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('output.avi', fourcc, 20.0, (width, height), isColor=False)
    while True:
        ret, frame = cap.read()
        frame = cv.flip(frame, 1)
        result = model(frame)
        frame = drawing_bbox(frame, result, model.names)
        
        cv.imshow('frame', frame)
        # cv.imshow('frame', gray)
        if cv.waitKey(1) == 27:
            break
finally:
    cap.release()
    out.release()
    cv.destroyAllWindows()