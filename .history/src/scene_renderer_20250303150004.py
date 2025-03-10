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
        scale = bbox_size / 100.0  # 增大缩放比例
        depth = 10  # 减小深度值，使车辆更容易看到
        
        # 创建基础变换矩阵
        transform = np.eye(4)
        
        # 设置位置
        transform[0, 3] = center[0] - image_shape[1]/2
        transform[1, 3] = -(center[1] - image_shape[0]/2)
        transform[2, 3] = -depth
        
        # 创建旋转矩阵（绕Z轴旋转，调整车辆朝向）
        rot_z = np.array([
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1]
        ])
        
        # 调整车辆初始朝向（因为模型的车头朝向y轴负方向）
        base_rot = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
        
        # 组合旋转
        final_rot = rot_z @ base_rot
        transform[:3, :3] = final_rot * scale
        
        return transform
        
    def render(self, image, detections):
        """渲染3D场景"""
        # 创建场景
        scene = pyrender.Scene(bg_color=[0.9, 0.9, 0.9, 1.0])
        
        # 设置相机
        camera = pyrender.IntrinsicsCamera(
            fx=1000,  # 使用固定焦距
            fy=1000,
            cx=image.shape[1]/2,
            cy=image.shape[0]/2,
            znear=0.1,
            zfar=1000.0
        )
        
        # 设置相机位置（俯视角度）
        camera_pose = np.array([
            [1, 0, 0, 0],
            [0, -0.7071, -0.7071, 20],  # 45度俯视角
            [0, 0.7071, -0.7071, 20],
            [0, 0, 0, 1]
        ])
        scene.add(camera, pose=camera_pose)
        
        # 添加多个光源
        # 主光源（顶部）
        main_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        main_light_pose = np.eye(4)
        main_light_pose[2, 3] = 20
        scene.add(main_light, pose=main_light_pose)
        
        # 前方光源
        front_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.5)
        front_light_pose = np.array([
            [1, 0, 0, 0],
            [0, 0.7071, -0.7071, 10],
            [0, 0.7071, 0.7071, 10],
            [0, 0, 0, 1]
        ])
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