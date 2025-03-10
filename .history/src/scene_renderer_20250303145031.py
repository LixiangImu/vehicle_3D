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
        
    def _calculate_vehicle_transform(self, front_point, rear_point, bbox_size):
        """
        计算车辆的变换矩阵，考虑俯视角度
        """
        # 计算车辆朝向角度
        direction = front_point - rear_point
        angle = -math.atan2(direction[1], direction[0])
        
        # 计算车辆中心点
        center = (front_point + rear_point) / 2
        
        # 调整缩放和深度
        scale = bbox_size / 200.0
        depth = 1000  # 增加深度值
        
        # 创建基础变换矩阵
        transform = np.eye(4)
        
        # 计算相机坐标系下的位置
        transform[0, 3] = (center[0] - self.camera_matrix[0,2]) * depth / self.camera_matrix[0,0]
        transform[1, 3] = (center[1] - self.camera_matrix[1,2]) * depth / self.camera_matrix[1,1]
        transform[2, 3] = -depth
        
        # 创建俯视角旋转矩阵（绕X轴旋转约45度）
        tilt_angle = math.radians(45)
        rot_x = np.array([
            [1, 0, 0],
            [0, math.cos(tilt_angle), -math.sin(tilt_angle)],
            [0, math.sin(tilt_angle), math.cos(tilt_angle)]
        ])
        
        # 创建平面旋转矩阵
        rot_z = np.array([
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1]
        ])
        
        # 调整模型初始朝向
        base_rot = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        
        # 组合所有旋转
        final_rot = rot_z @ rot_x @ base_rot
        transform[:3, :3] = final_rot * scale
        
        return transform
        
    def render(self, image, detections):
        """
        渲染3D场景
        """
        # 创建场景
        scene = pyrender.Scene(bg_color=[0.9, 0.9, 0.9, 1.0])
        
        # 设置相机
        camera = pyrender.IntrinsicsCamera(
            fx=self.camera_matrix[0,0],
            fy=self.camera_matrix[1,1],
            cx=self.camera_matrix[0,2],
            cy=self.camera_matrix[1,2],
            znear=100.0,
            zfar=2000.0
        )
        
        # 设置相机俯视角度
        camera_pose = np.array([
            [1, 0, 0, 0],
            [0, math.cos(math.radians(-45)), -math.sin(math.radians(-45)), 500],
            [0, math.sin(math.radians(-45)), math.cos(math.radians(-45)), -1000],
            [0, 0, 0, 1]
        ])
        scene.add(camera, pose=camera_pose)
        
        # 添加多个光源
        # 主光源（顶部）
        main_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        main_light_pose = np.eye(4)
        main_light_pose[1, 3] = 1000
        scene.add(main_light, pose=main_light_pose)
        
        # 辅助光源（前方）
        front_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        front_light_pose = np.eye(4)
        front_light_pose[2, 3] = 1000
        scene.add(front_light, pose=front_light_pose)
        
        # 渲染车辆
        for detection in detections:
            bbox = detection['bbox']
            front_point = detection['front_point']
            rear_point = detection['rear_point']
            
            bbox_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
            transform = self._calculate_vehicle_transform(
                np.array(front_point),
                np.array(rear_point),
                bbox_size
            )
            
            mesh = pyrender.Mesh.from_trimesh(self.vehicle_mesh, smooth=True)
            scene.add(mesh, pose=transform)
        
        # 渲染场景
        r = pyrender.OffscreenRenderer(image.shape[1], image.shape[0])
        color, _ = r.render(scene)
        
        return cv2.cvtColor(color, cv2.COLOR_RGB2BGR) 