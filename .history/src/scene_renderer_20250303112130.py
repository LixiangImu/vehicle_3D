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
        # 计算车辆朝向角度
        direction = front_point - rear_point
        angle = math.atan2(direction[1], direction[0])
        
        # 计算车辆中心点
        center = (front_point + rear_point) / 2
        
        # 估算深度（这里使用简化的方法）
        depth = 1000.0  # 可以根据实际情况调整
        
        # 根据边界框大小估算缩放比例
        scale = bbox_size / 100.0
        
        # 创建变换矩阵
        transform = np.eye(4)
        transform[:3, 3] = [center[0], center[1], depth]
        
        # 创建旋转矩阵
        rot_z = np.array([
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1]
        ])
        
        # 添加90度旋转使模型朝向与y轴负方向对齐
        rot_x = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        
        final_rot = rot_z @ rot_x
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
        # 创建渲染场景，使用白色背景
        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0])
        
        # 添加相机
        camera = pyrender.IntrinsicsCamera(
            fx=self.camera_matrix[0,0],
            fy=self.camera_matrix[1,1],
            cx=self.camera_matrix[0,2],
            cy=self.camera_matrix[1,2]
        )
        scene.add(camera)
        
        # 添加多个方向光源以更好地显示白模型
        light1 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        light2 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.5)
        light3 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.5)
        
        # 从不同角度添加光源
        scene.add(light1, pose=np.eye(4))
        
        light2_pose = np.eye(4)
        light2_pose[:3, 3] = [0, -1, 1]
        scene.add(light2, pose=light2_pose)
        
        light3_pose = np.eye(4)
        light3_pose[:3, 3] = [0, 1, 1]
        scene.add(light3, pose=light3_pose)
        
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
        
        # 转换为OpenCV格式（BGR）
        render_result = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        
        return render_result 