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
        计算车辆的变换矩阵
        Args:
            front_point: 车辆前点坐标
            rear_point: 车辆后点坐标
            bbox_size: 边界框大小
        Returns:
            4x4变换矩阵
        """
        # 计算车辆朝向角度（图像坐标系中）
        direction = front_point - rear_point
        angle = math.atan2(direction[1], direction[0])
        
        # 计算车辆中心点
        center = (front_point + rear_point) / 2
        
        # 设置合适的深度和缩放
        depth = bbox_size * 1.5
        scale = bbox_size / 100.0
        
        # 创建变换矩阵
        transform = np.eye(4)
        
        # 设置平移（注意图像坐标系到3D坐标系的转换）
        transform[0, 3] = center[0]  # x
        transform[1, 3] = center[1]  # y
        transform[2, 3] = -depth     # z（负值使物体在相机前方）
        
        # 创建旋转矩阵
        # 1. 首先绕Z轴旋转使车辆朝向正确的方向
        rot_z = np.array([
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1]
        ])
        
        # 2. 调整车辆初始朝向（因为模型的车头朝向y轴负方向）
        init_rot = np.array([
            [0, -1, 0],   # x轴方向
            [1, 0, 0],    # y轴方向
            [0, 0, 1]     # z轴方向
        ])
        
        # 3. 调整为相机坐标系（使车辆正面朝向相机）
        cam_rot = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        
        # 组合所有旋转
        final_rot = rot_z @ init_rot @ cam_rot
        
        # 应用旋转和缩放
        transform[:3, :3] = final_rot * scale
        
        return transform
        
    def render(self, image, detections):
        """
        渲染3D场景
        Args:
            image: 输入图像（仅用于获取尺寸）
            detections: 检测结果列表
        Returns:
            渲染后的纯3D场景图像
        """
        # 创建渲染场景，使用灰色背景以便于观察
        scene = pyrender.Scene(bg_color=[0.8, 0.8, 0.8, 1.0])
        
        # 添加相机，调整相机参数
        camera = pyrender.IntrinsicsCamera(
            fx=self.camera_matrix[0,0],
            fy=self.camera_matrix[1,1],
            cx=self.camera_matrix[0,2],
            cy=self.camera_matrix[1,2],
            znear=0.05,
            zfar=1000.0
        )
        
        # 添加相机节点，设置相机姿态
        camera_pose = np.eye(4)
        camera_pose[2, 3] = 0  # 将相机放在原点
        scene.add(camera, pose=camera_pose)
        
        # 添加更强的光照
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)
        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, 0, 1]  # 将光源放在相机前方
        scene.add(light, pose=light_pose)
        
        # 处理每个检测到的车辆
        for detection in detections:
            bbox = detection['bbox']
            front_point = detection['front_point']
            rear_point = detection['rear_point']
            
            # 计算边界框大小
            bbox_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
            
            # 计算变换矩阵
            transform = self._calculate_vehicle_transform(
                np.array(front_point),
                np.array(rear_point),
                bbox_size
            )
            
            # 添加车辆模型到场景
            mesh = pyrender.Mesh.from_trimesh(self.vehicle_mesh, smooth=True)
            scene.add(mesh, pose=transform)
        
        # 渲染场景
        r = pyrender.OffscreenRenderer(image.shape[1], image.shape[0])
        color, depth = r.render(scene)
        
        # 转换为OpenCV格式
        render_result = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        
        return render_result 