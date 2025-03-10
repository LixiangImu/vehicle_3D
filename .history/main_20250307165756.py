import cv2
import numpy as np
import torch
from ultralytics import YOLO
import open3d as o3d
from utils.camera_utils import *
import config

class VehicleDetector3D:
    def __init__(self):
        # 加载YOLO模型
        self.model = YOLO(config.YOLO_MODEL_PATH)
        
        # 初始化相机参数
        self.camera_matrix = config.CAMERA_MATRIX
        self.dist_coeffs = config.DIST_COEFFS
        
        # 如果需要降采样，调整相机矩阵
        if config.DOWNSAMPLE_FACTOR > 1:
            self.camera_matrix = resize_camera_matrix(
                self.camera_matrix, 
                config.DOWNSAMPLE_FACTOR
            )
    
    def process_image(self, image):
        # 降采样
        if config.DOWNSAMPLE_FACTOR > 1:
            width = image.shape[1] // config.DOWNSAMPLE_FACTOR
            height = image.shape[0] // config.DOWNSAMPLE_FACTOR
            image = cv2.resize(image, (width, height))
        
        # 运行YOLO检测
        results = self.model(image)
        
        # 处理检测结果
        vehicles = []
        for result in results[0].boxes:
            bbox = result.xyxy[0].cpu().numpy()  # 获取边界框
            keypoints = result.keypoints[0].cpu().numpy()  # 获取关键点
            
            # 提取前后点坐标
            front_point = keypoints[0][:2]  # 前点
            rear_point = keypoints[1][:2]   # 后点
            
            # 估计3D位姿
            success, rotation_vector, translation_vector = estimate_vehicle_pose(
                front_point,
                rear_point,
                self.camera_matrix,
                self.dist_coeffs,
                config.VEHICLE_LENGTH
            )
            
            if success:
                vehicles.append({
                    'bbox': bbox,
                    'front_point': front_point,
                    'rear_point': rear_point,
                    'rotation': rotation_vector,
                    'translation': translation_vector
                })
        
        return vehicles

    def visualize_results(self, image, vehicles):
        # 暂时只显示2D检测结果
        vis_img = image.copy()
        for vehicle in vehicles:
            bbox = vehicle['bbox'].astype(int)
            cv2.rectangle(vis_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # 绘制关键点
            cv2.circle(vis_img, tuple(vehicle['front_point'].astype(int)), 5, (255, 0, 0), -1)
            cv2.circle(vis_img, tuple(vehicle['rear_point'].astype(int)), 5, (0, 0, 255), -1)
        
        return vis_img

def main():
    detector = VehicleDetector3D()
    
    # 读取测试图像
    image = cv2.imread('test_image.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 处理图像
    vehicles = detector.process_image(image)
    
    # 显示结果
    vis_img = detector.visualize_results(image, vehicles)
    cv2.imshow('Detection Results', cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
