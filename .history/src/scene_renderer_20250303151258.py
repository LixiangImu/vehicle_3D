import numpy as np
import cv2
import trimesh
import pyrender
import math
from PIL import Image

class SceneRenderer:
    def __init__(self, camera_calib_file, model_path):
        """
        初始化场景渲染器
        Args:
            camera_calib_file: 相机标定文件路径
            model_path: 3D模型文件路径
        """
        self.camera_matrix, self.dist_coeffs = self._load_camera_calibration(camera_calib_file)
        self.vehicle_mesh = trimesh.load(model_path)
        
    def _load_camera_calibration(self, calib_file):
        """加载相机标定数据"""
        data = np.load(calib_file)
        return data['camera_matrix'], data['dist_coeffs']
        
    def _calculate_vehicle_transform(self, front_point, rear_point, bbox_size, image_shape):
        """计算车辆的变换矩阵"""
        # 计算车辆朝向角度
        direction = front_point - rear_point
        angle = -math.atan2(direction[1], direction[0])
        
        # 计算车辆中心点
        center = (front_point + rear_point) / 2
        
        # 调整缩放和深度
        scale = bbox_size / 150.0  # 调整缩放比例
        depth = 100  # 减小深度值，使车辆更容易看到
        
        # 创建基础变换矩阵
        transform = np.eye(4)
        
        # 设置位置（使用相机内参进行投影）
        transform[0, 3] = (center[0] - image_shape[1]/2) * depth / self.camera_matrix[0,0]
        transform[1, 3] = (center[1] - image_shape[0]/2) * depth / self.camera_matrix[1,1]
        transform[2, 3] = -depth
        
        # 创建旋转矩阵（绕Z轴旋转，调整车辆朝向）
        rot_z = np.array([
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1]
        ])
        
        # 调整车辆初始朝向（y轴负方向为车头）
        base_rot = np.array([
            [0, -1, 0],  # x轴方向（车辆左侧）
            [-1, 0, 0],  # y轴方向（车头朝向）
            [0, 0, -1]   # z轴方向（车辆高度）
        ])
        
        # 组合旋转
        final_rot = rot_z @ base_rot
        transform[:3, :3] = final_rot * scale
        
        return transform
        
    def render(self, image, detections):
        """渲染3D场景"""
        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0])  # 使用纯白背景
        
        # 设置相机
        camera = pyrender.IntrinsicsCamera(
            fx=self.camera_matrix[0,0],
            fy=self.camera_matrix[1,1],
            cx=self.camera_matrix[0,2],
            cy=self.camera_matrix[1,2],
            znear=1.0,
            zfar=2000.0  # 增加远平面距离
        )
        
        # 设置相机位置（俯视角度约45度，距离更远）
        camera_pose = np.array([
            [1, 0, 0, 0],
            [0, 0.7071, -0.7071, 0],   # 45度俯视角
            [0, 0.7071, 0.7071, 500],  # 增加相机高度到500
            [0, 0, 0, 1]
        ])
        scene.add(camera, pose=camera_pose)
        
        # 添加多个光源以改善照明
        # 主光源（顶部）
        main_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        main_light_pose = np.eye(4)
        main_light_pose[2, 3] = 500
        scene.add(main_light, pose=main_light_pose)
        
        # 侧面光源
        side_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        side_light_pose = np.array([
            [1, 0, 0, 300],
            [0, 1, 0, 0],
            [0, 0, 1, 300],
            [0, 0, 0, 1]
        ])
        scene.add(side_light, pose=side_light_pose)
        
        # 渲染车辆
        for detection in detections:
            bbox = detection['bbox']
            front_point = detection['front_point']
            rear_point = detection['rear_point']
            
            bbox_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
            transform = self._calculate_vehicle_transform(
                np.array(front_point),
                np.array(rear_point),
                bbox_size,
                image.shape[:2]
            )
            
            try:
                mesh = pyrender.Mesh.from_trimesh(self.vehicle_mesh, smooth=True)
                scene.add(mesh, pose=transform)
            except Exception as e:
                print(f"添加模型时出错: {e}")
        
        # 渲染场景
        r = pyrender.OffscreenRenderer(image.shape[1], image.shape[0])
        color, depth = r.render(scene)
        
        return cv2.cvtColor(color, cv2.COLOR_RGB2BGR) 