import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
import math
import torch
import copy
from ultralytics import YOLO  # 添加这行导入

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

    def detect_vehicles(self, image):
        """
        使用YOLOv8模型检测车辆
        Args:
            image: 输入图像 (numpy array)
        Returns:
            detections: 列表，每个元素包含 [bbox, front_point, rear_point]
        """
        # 加载YOLOv8模型
        model = YOLO('pose.pt')
        
        # 预处理图像并进行检测
        results = model(image)
        
        # 解析检测结果
        detections = []
        for result in results[0]:  # 遍历每个检测结果
            # 获取边界框
            bbox = result.boxes.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            
            # 获取关键点坐标
            keypoints = result.keypoints.xy[0].cpu().numpy()  # 所有关键点坐标
            front_point = keypoints[0]  # 前端关键点 [x, y]
            rear_point = keypoints[1]   # 后端关键点 [x, y]
            
            detections.append([bbox, front_point, rear_point])
        
        return detections

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
        # 将输入点转换为正确的格式
        image_point = np.array([[image_point]], dtype=np.float32)
        
        # 使用相机标定参数进行坐标转换
        normalized_point = cv2.undistortPoints(
            image_point, 
            self.camera_matrix, 
            self.dist_coeffs
        )[0][0]
        
        # 修改射线与地平面的交点计算方法
        focal_length = self.camera_matrix[0,0]  # 假设fx和fy相近
        scale = focal_length / (normalized_point[1] + 1e-6)  # 防止除零
        world_x = normalized_point[0] * scale
        world_z = scale
        world_y = self.ground_height
        
        return np.array([world_x, world_y, world_z])

    def estimate_vehicle_pose(self, front_point, rear_point, bbox):
        """
        估计车辆在3D空间中的位姿
        """
        # 转换前后点到世界坐标
        front_3d = self.image_to_world_coordinates(front_point)
        rear_3d = self.image_to_world_coordinates(rear_point)
        
        # 计算车辆朝向（修改角度计算）
        direction = front_3d - rear_3d
        angle = np.arctan2(direction[0], direction[2])  # 修改这里的角度计算
        
        # 计算车辆中心点
        center_3d = (front_3d + rear_3d) / 2
        
        # 根据边界框估计车辆尺寸
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        image_diagonal = np.sqrt(bbox_width**2 + bbox_height**2)
        
        # 使用边界框对角线长度来估计缩放因子
        scale_factor = (image_diagonal / 100.0) * (np.linalg.norm(front_3d - rear_3d) / self.default_car_length)
        
        return center_3d, angle, scale_factor

    def visualize_3d_scene(self, detections):
        """
        可视化3D场景并返回渲染图像
        """
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)  # 设置为不可见窗口
        
        # 添加坐标系
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=2.0, origin=[0, 0, 0])  # 增大坐标系大小
        vis.add_geometry(coordinate_frame)
        
        # 处理每个检测到的车辆
        for detection in detections:
            bbox, front_point, rear_point = detection
            
            # 获取车辆位姿
            center, angle, scale = self.estimate_vehicle_pose(
                front_point, rear_point, bbox)
            
            # 复制并变换3D模型
            car_model_transformed = copy.deepcopy(self.car_model)
            
            # 先进行缩放
            car_model_transformed.scale(scale, center=car_model_transformed.get_center())
            
            # 然后进行旋转
            R = car_model_transformed.get_rotation_matrix_from_xyz((0, angle, 0))
            car_model_transformed.rotate(R, center=car_model_transformed.get_center())
            
            # 最后进行平移
            car_model_transformed.translate(center)
            
            # 为每个车辆模型设置不同的颜色
            car_model_transformed.paint_uniform_color([np.random.random(), 
                                                     np.random.random(), 
                                                     np.random.random()])
            
            vis.add_geometry(car_model_transformed)
        
        # 设置视角和渲染选项
        vis.get_render_option().background_color = np.asarray([0.8, 0.8, 0.8])
        vis.get_render_option().point_size = 1
        
        # 设置更好的视角
        ctr = vis.get_view_control()
        ctr.set_zoom(0.3)
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 1, 0])
        
        # 渲染场景
        vis.poll_events()
        vis.update_renderer()
        
        # 捕获渲染图像
        rendered_image = np.asarray(vis.capture_screen_float_buffer())
        
        # 关闭窗口
        vis.destroy_window()
        
        return rendered_image

def main():
    # 初始化参数
    camera_calib_path = "camera_calibration.npz"
    model_path = r"D:\Desktop\yitaixx\3D_whiteModel\3Dmodel\car.obj"
    image_path = r"D:\Desktop\yitaixx\3D_whiteModel\data\car\car2.jpg"
    
    # 创建检测器实例
    detector = VehicleDetector3D(camera_calib_path, model_path)
    
    # 读取图像并检测车辆
    image = cv2.imread(image_path)
    detections = detector.detect_vehicles(image)
    
    # 可视化结果
    rendered_image = detector.visualize_3d_scene(detections)
    cv2.imshow('Rendered Image', rendered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()