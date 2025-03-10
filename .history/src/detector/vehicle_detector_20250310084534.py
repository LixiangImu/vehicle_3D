import numpy as np
from ultralytics import YOLO
from ..utils.geometry import ImageToWorldConverter
from .pose_estimator import VehiclePoseEstimator

class VehicleDetector:
    def __init__(self, config):
        """
        初始化车辆检测器
        Args:
            config: 配置对象
        """
        self.config = config
        self.model = YOLO('pose.pt')
        self.converter = ImageToWorldConverter(config)
        self.pose_estimator = VehiclePoseEstimator(config)

    def detect(self, image):
        """
        检测图像中的车辆
        Args:
            image: 输入图像
        Returns:
            detections: 检测结果列表
        """
        results = self.model(image)
        detections = []
        
        for result in results[0]:
            bbox = result.boxes.xyxy[0].cpu().numpy()
            keypoints = result.keypoints.xy[0].cpu().numpy()
            front_point = keypoints[0]
            rear_point = keypoints[1]
            detections.append({
                'bbox': bbox,
                'front_point': front_point,
                'rear_point': rear_point
            })
            
        return detections 