import cv2
import numpy as np
import torch
from ultralytics import YOLO
import open3d as o3d
from utils.camera_utils import *
import config
from utils.visualization import Scene3D
from config import IMAGE_HEIGHT, IMAGE_WIDTH

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
        
        # 初始化3D场景
        self.scene = Scene3D(config.VEHICLE_MODEL_PATH)
        self.scene.update_camera_params(
            self.camera_matrix,
            (IMAGE_HEIGHT // config.DOWNSAMPLE_FACTOR, 
             IMAGE_WIDTH // config.DOWNSAMPLE_FACTOR)
        )
    
    def estimate_camera_extrinsics(self, vehicles):
        """估计相机外参（简化版本）"""
        if not vehicles:
            return None
        
        # 使用第一辆车的位置来估计相机高度和俯仰角
        first_vehicle = vehicles[0]
        translation = first_vehicle['translation']
        
        # 假设相机正对着车辆中心
        camera_height = abs(translation[2]) + 1.5  # 假设相机比车高1.5米
        
        # 计算相机外参矩阵
        pitch_angle = np.arctan2(camera_height, abs(translation[2]))
        
        extrinsic = np.array([
            [1, 0, 0, 0],
            [0, np.cos(pitch_angle), -np.sin(pitch_angle), 0],
            [0, np.sin(pitch_angle), np.cos(pitch_angle), -camera_height],
            [0, 0, 0, 1]
        ])
        
        return extrinsic
    
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
        for result in results:
            boxes = result.boxes  # 获取所有检测框
            keypoints = result.keypoints  # 获取所有关键点
            
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()  # 获取边界框
                conf = boxes.conf[i].cpu().numpy()  # 获取置信度
                kpts = keypoints.xy[i].cpu().numpy()  # 获取关键点坐标
                
                # 提取前后点坐标
                front_point = kpts[0]  # 前点
                rear_point = kpts[1]   # 后点
                
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
                        'confidence': conf,
                        'front_point': front_point,
                        'rear_point': rear_point,
                        'rotation': rotation_vector,
                        'translation': translation_vector
                    })
        
        # 估计相机外参
        camera_extrinsic = self.estimate_camera_extrinsics(vehicles)
        if camera_extrinsic is not None:
            # 更新场景中的相机参数
            self.scene.update_camera_params(self.camera_matrix, image.shape[:2])
        
        # 更新3D场景
        self.scene.update_scene(vehicles)
        
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

    def __del__(self):
        if hasattr(self, 'scene'):
            self.scene.close()

def main():
    detector = VehicleDetector3D()
    
    # 读取图像文件
    image_path = r"data\car\car2.jpg" 
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    try:
        # 转换为RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 处理图像
        vehicles = detector.process_image(frame_rgb)
        
        # 显示2D检测结果
        vis_img = detector.visualize_results(frame_rgb, vehicles)
        cv2.imshow('Detection Results', cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        
        # 等待按键
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cv2.destroyAllWindows()
        detector.scene.close()

if __name__ == "__main__":
    main()
