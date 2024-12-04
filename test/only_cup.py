import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import openvino as ov
import requests
import torch
from pathlib import Path
from typing import Tuple

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # yolov5s 모델 사용


# Fetch `notebook_utils` module
r = requests.get(
    url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py", "w").write(r.text)
import notebook_utils as utils

# A directory where the model will be downloaded.
base_model_dir = Path("model")

device = utils.device_widget()
device

# Initialize OpenVINO Runtime runtime.
core = ov.Core()

def model_init(model_path: str) -> Tuple:
    model = core.read_model(model=model_path)
    compiled_model = core.compile_model(model=model, device_name=device.value)
    input_keys = compiled_model.input(0)
    output_keys = compiled_model.output(0)
    return input_keys, output_keys, compiled_model

beverage_classes = ["cup", "can", "bottle"]

# YOLO 모델을 사용하여 음료 객체를 감지
def detect_beverages(frame):
    results = model(frame)  # 이미지에서 객체 검출
    detections = results.xyxy[0].cpu().numpy()  # x1, y1, x2, y2, confidence, class

    # 감지된 객체가 음료 관련 클래스인 경우만 필터링
    beverage_detections = []
    for *box, conf, cls in detections:
        if model.names[int(cls)] in beverage_classes:  # 음료 관련 클래스일 경우
            beverage_detections.append([*box, conf, cls])
    
    return beverage_detections

# 음료 객체의 결과를 이미지에 그리는 함수
def draw_beverages(frame, detections):
    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        label = f"{model.names[int(cls)]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 바운딩 박스 그리기
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # 레이블 텍스트 추가
    return frame

# 비디오 파일을 처리하는 함수
def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        exit()

    # 비디오 저장 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (640, 480))  # Adjust frame size as needed

    # Read frames from the video
    while cap.isOpened():
        ret, frame = cap.read()  # 비디오 프레임 읽기
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # 음료 객체 감지
        detections = detect_beverages(frame)
        frame = draw_beverages(frame, detections)

        # 객체 검출 결과 화면에 표시
        cv2.imshow("Beverage Detection", frame)
        
        # 결과 비디오에 저장
        out.write(frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# 입력 비디오 파일 경로와 출력 파일 경로 설정
video_path = '/home/intel/openvino/beverage.mp4'  # 비디오 파일 경로
output_path = 'output/processed_video.mp4'  # 출력 비디오 경로

process_video(video_path, output_path)
