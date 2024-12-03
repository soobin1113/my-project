import os
import cv2
import numpy as np
import torch
import time
import requests
import openvino as ov
from pathlib import Path
from typing import Tuple
import matplotlib.cm

# 모델 다운로드 및 설정 (YOLOv5)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # yolov5s 모델 사용

# Fetch `notebook_utils` module
r = requests.get(
    url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)
open("notebook_utils.py", "w").write(r.text)
import notebook_utils as utils

# A directory where the model will be downloaded.
base_model_dir = Path("model")

device = utils.device_widget()  # device selection widget
device

# Initialize OpenVINO Runtime runtime.
core = ov.Core()

# 모델 초기화 함수 (OpenVINO 모델)
def model_init(model_path: str) -> Tuple:
    model = core.read_model(model=model_path)
    compiled_model = core.compile_model(model=model, device_name=device.value)
    input_keys = compiled_model.input(0)
    output_keys = compiled_model.output(0)
    return input_keys, output_keys, compiled_model

beverage_classes = ["cup", "can", "bottle"]

# WebcamProcessor 클래스
class WebcamProcessor:
    def __init__(self, camera_id=0, frame_width=1280, frame_height=720):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise ValueError("웹캠을 열 수 없습니다.")
        
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self.current_frame = None

    def read_frame(self):
        """웹캠으로부터 프레임을 읽어옵니다."""
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("웹캠에서 영상을 읽을 수 없습니다.")
        self.current_frame = frame
        return frame

    def release(self):
        """웹캠 자원을 해제합니다."""
        self.cap.release()

# DepthProcessor 클래스
class DepthProcessor:
    def __init__(self, compiled_model, input_key, output_key):
        self.compiled_model = compiled_model
        self.input_key = input_key
        self.output_key = output_key

    def process_frame(self, frame):
        """주어진 프레임에서 뎁스 결과를 생성합니다."""
        resized_frame = cv2.resize(frame, (self.input_key.shape[2], self.input_key.shape[3]))
        input_image = np.expand_dims(np.transpose(resized_frame, (2, 0, 1)), 0)
        result = self.compiled_model([input_image])[self.output_key]
        return result

    def visualize_result(self, result):
        """뎁스 결과를 시각화합니다."""
        result_frame = self.convert_result_to_image(result)
        return result_frame

    @staticmethod
    def normalize_minmax(data):
        """뎁스 데이터를 정규화합니다."""
        return (data - data.min()) / (data.max() - data.min())

    def convert_result_to_image(self, result, colormap="viridis"):
        """뎁스 결과를 컬러맵으로 변환합니다."""
        cmap = matplotlib.cm.get_cmap(colormap)
        result = result.squeeze(0)
        result = self.normalize_minmax(result)
        result = cmap(result)[:, :, :3] * 255
        result = result.astype(np.uint8)
        return result

# YOLO 모델을 사용하여 음료 객체를 감지하는 함수
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

# 섹션을 나누어 심도 맵을 처리하고, 왼쪽과 오른쪽으로 나누는 함수
def process_depth_sections(depth_map, num_rows=4, num_cols=4, threshold=0.8):
    h, w = depth_map.shape
    section_height = h // num_rows
    section_width = w // num_cols
    
    left_count = 0
    right_count = 0

    for row in range(num_rows):
        for col in range(num_cols):
            # 섹션 좌표 계산
            y1, y2 = row * section_height, (row + 1) * section_height
            x1, x2 = col * section_width, (col + 1) * section_width

            # 섹션 영역 추출
            section = depth_map[y1:y2, x1:x2]
            
            # 섹션의 평균 심도 계산
            mean_depth = section.mean()
            
            # 기준값 이상인지 확인
            if mean_depth >= threshold:
                if col < num_cols // 2:
                    left_count += 1  # 왼쪽 영역
                else:
                    right_count += 1  # 오른쪽 영역

    # 결정 논리
    if left_count > right_count:
        return "Avoid to Right"
    elif right_count > left_count:
        return "Avoid to Left"
    else:
        return "Balanced"

# 16등분 영역에 심도 정보 표시 함수
def display_depth_sections(image, depth_map, num_rows=4, num_cols=4, output_width=1280, output_height=720):
    # 출력 해상도로 크기 조정
    image = cv2.resize(image, (output_width, output_height))
    depth_map = cv2.resize(depth_map, (output_width, output_height))

    # 섹션 크기 계산
    section_height = output_height // num_rows
    section_width = output_width // num_cols

    for row in range(num_rows):
        for col in range(num_cols):
            # 섹션 좌표 계산
            y1, y2 = row * section_height, (row + 1) * section_height
            x1, x2 = col * section_width, (col + 1) * section_width

            # 섹션 영역 추출
            section = depth_map[y1:y2, x1:x2]
            
            # 섹션의 평균 심도 계산
            mean_depth = section.mean()

            # 섹션에 심도 값 표시
            cv2.putText(
                image,
                f"{mean_depth:.2f}",
                (x1 + 10, y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),  # 흰색 텍스트
                1,
                cv2.LINE_AA
            )
            
            # 섹션 경계선 그리기
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)  # 초록색 경계선

    return image

# 메인 루프
def main():
    # 웹캠 및 모델 초기화
    webcam = WebcamProcessor(camera_id=0)  # 웹캠 ID 0 설정
    
    # OpenVINO 모델 초기화
    core = ov.Core()
    model_folder = Path("model")
    model_xml_path = model_folder / "MiDaS_small.xml"
    compiled_model = model_init(str(model_xml_path))[2]
    input_key = compiled_model.input(0)
    output_key = compiled_model.output(0)
    
    depth_processor = DepthProcessor(compiled_model, input_key, output_key)

    try:
        while True:
            frame = webcam.read_frame()  # 웹캠에서 프레임 읽기
            depth_result = depth_processor.process_frame(frame)  # 뎁스 처리
            
            # 뎁스 맵 정규화
            depth_map = (depth_result.squeeze(0) - depth_result.min()) / (depth_result.max() - depth_result.min())
            
            # 16등분 섹션 처리 및 결정
            decision = process_depth_sections(depth_map, num_rows=4, num_cols=4, threshold=0.8)
            
            # 시각화된 뎁스 맵
            depth_frame = depth_processor.visualize_result(depth_result)
            
            # 16등분 영역에 심도 정보 표시
            display_frame = display_depth_sections(depth_frame.copy(), depth_map, num_rows=4, num_cols=4, output_width=1280, output_height=720)
            
            # 음료 객체 감지
            detections = detect_beverages(frame)
            display_frame = draw_beverages(display_frame, detections)
            
            # 결정 결과 출력
            cv2.putText(
                display_frame,
                decision,
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),  # 빨간색 텍스트
                2,
                cv2.LINE_AA
            )
            
            # 실시간 출력
            cv2.imshow("Depth Estimation with Beverage Detection", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키로 종료
                break
    except KeyboardInterrupt:
        print("처리가 중단되었습니다.")
    finally:
        webcam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
