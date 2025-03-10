import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
import math
import torch

class VehicleDetector3D:
    def __init__(self, camera_calib_path, model_path):
        """
        初始化类
        Args:
            camera_calib_path: 相机标定文件路径
            model_path: 3D车辆模型文件路径
        """
        # 加载相机标定数据
        calib_data = np.load(camera_calib_path)
        self.camera_matrix = calib_data['camera_matrix']
        self.dist_coeffs = calib_data['dist_coeffs']
        
        # 加载3D白模型
        self.car_model = o3d.io.read_triangle_mesh(model_path)
        
        # 定义地平面高度（假设为0）
        self.ground_height = 0
        
        # 预设的平均车辆尺寸（单位：米）
        self.default_car_length = 4.5
        self.default_car_width = 1.8
        self.default_car_height = 1.5

    def detect_vehicles(self, image_path):
        """
        使用YOLO模型检测车辆
        返回：边界框和关键点列表
        """
        # 这里应该是您的YOLO模型检测代码
        # 为演示目的，使用模拟数据
        return self._simulate_detections()

    def _simulate_detections(self):
        """
        生成模拟的检测数据用于测试
        """
        # 模拟数据：[[bbox], [前端关键点], [后端关键点]]
        return [
            [[100, 100, 300, 200], [150, 120], [250, 120]],
            [[400, 150, 600, 250], [450, 170], [550, 170]]
        ]

    def image_to_world_coordinates(self, image_point):
        """
        将图像坐标转换为世界坐标
        """
        # 使用相机标定参数进行坐标转换
        normalized_point = cv2.undistortPoints(
            np.array([image_point]), 
            self.camera_matrix, 
            self.dist_coeffs
        )[0][0]
        
        # 假设地平面Z=0，计算射线与地平面的交点
        scale = -self.camera_matrix[2,2] / normalized_point[1]
        world_x = normalized_point[0] * scale
        world_y = self.ground_height
        world_z = scale
        
        return np.array([world_x, world_y, world_z])

    def estimate_vehicle_pose(self, front_point, rear_point, bbox):
        """
        估计车辆在3D空间中的位姿
        """
        # 转换前后点到世界坐标
        front_3d = self.image_to_world_coordinates(front_point)
        rear_3d = self.image_to_world_coordinates(rear_point)
        
        # 计算车辆朝向
        direction = front_3d - rear_3d
        angle = np.arctan2(direction[2], direction[0])
        
        # 计算车辆中心点
        center_3d = (front_3d + rear_3d) / 2
        
        # 估计车辆尺寸
        length = np.linalg.norm(front_3d - rear_3d)
        scale_factor = length / self.default_car_length
        
        return center_3d, angle, scale_factor

    def visualize_3d_scene(self, detections):
        """
        可视化3D场景
        """
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        # 添加坐标系
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0, origin=[0, 0, 0])
        vis.add_geometry(coordinate_frame)
        
        # 处理每个检测到的车辆
        for detection in detections:
            bbox, front_point, rear_point = detection
            
            # 获取车辆位姿
            center, angle, scale = self.estimate_vehicle_pose(
                front_point, rear_point, bbox)
            
            # 复制并变换3D模型
            car_model_transformed = copy.deepcopy(self.car_model)
            R = car_model_transformed.get_rotation_matrix_from_xyz((0, angle, 0))
            car_model_transformed.rotate(R, center=True)
            car_model_transformed.scale(scale, center=True)
            car_model_transformed.translate(center)
            
            vis.add_geometry(car_model_transformed)
        
        # 设置视角和渲染选项
        vis.get_render_option().background_color = np.asarray([0.8, 0.8, 0.8])
        vis.get_render_option().point_size = 1
        vis.run()
        vis.destroy_window()

def main():
    # 初始化参数
    camera_calib_path = "camera_calibration.npz"
    model_path = r"D:\Desktop\yitaixx\3D_whiteModel\3Dmodel\car.obj"
    image_path = "D:\Desktop\yitaixx\3D_whiteModel\data\car\car2.jpg"
    
    # 创建检测器实例
    detector = VehicleDetector3D(camera_calib_path, model_path)
    
    # 读取图像并检测车辆
    image = cv2.imread(image_path)
    detections = detector.detect_vehicles(image)
    
    # 可视化结果
    detector.visualize_3d_scene(detections)

if __name__ == "__main__":
    main()