import torch
from ultralytics import YOLO
import numpy as np

class VehicleDetector:
    def __init__(self, weights_path):
        """
        初始化车辆检测器
        Args:
            weights_path: YOLO权重文件路径
        """
        self.model = YOLO(weights_path)
        
    def detect(self, image):
        """
        检测图像中的车辆
        """
        results = self.model(image)[0]
        detections = []
        
        boxes = results.boxes.data
        keypoints = results.keypoints.data
        
        for i, result in enumerate(boxes):
            if len(result) >= 6:  # 确保有足够的数据
                confidence = float(result[4])
                if confidence < 0.3:  # 置信度阈值
                    continue
                    
                # 获取边界框
                bbox = result[:4].cpu().numpy()
                
                # 获取当前车辆的关键点
                if keypoints is not None and i < len(keypoints):
                    vehicle_keypoints = keypoints[i].cpu().numpy()
                    front_point = vehicle_keypoints[0][:2]  # 前点
                    rear_point = vehicle_keypoints[1][:2]   # 后点
                    
                    detections.append({
                        'bbox': bbox,
                        'front_point': front_point,
                        'rear_point': rear_point,
                        'confidence': confidence
                    })
        
        return detections 