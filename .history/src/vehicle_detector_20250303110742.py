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
        Args:
            image: OpenCV格式的图像(BGR)
        Returns:
            list of dict: 每个车辆的检测结果，包含边界框和关键点
        """
        results = self.model(image)[0]
        detections = []
        
        for result in results.boxes.data:
            if len(result) >= 6:  # 确保有足够的数据
                confidence = float(result[4])
                if confidence < 0.3:  # 置信度阈值
                    continue
                    
                # 获取边界框
                bbox = result[:4].cpu().numpy()
                
                # 获取关键点（假设模型输出包含前后点）
                keypoints = results.keypoints.data[0].cpu().numpy()
                front_point = keypoints[0][:2]  # 前点
                rear_point = keypoints[1][:2]   # 后点
                
                detections.append({
                    'bbox': bbox,
                    'front_point': front_point,
                    'rear_point': rear_point,
                    'confidence': confidence
                })
                
        return detections 