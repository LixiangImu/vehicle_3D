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
        """
        # 计算车辆朝向角度
        direction = front_point - rear_point
        angle = -math.atan2(direction[1], direction[0])
        
        # 计算车辆中心点
        center = (front_point + rear_point) / 2
        
        # 调整缩放和深度
        scale = bbox_size / 400.0  # 减小缩放比例
        depth = 100  # 减小深度值，使车辆更靠近相机
        
        # 创建基础变换矩阵
        transform = np.eye(4)
        
        # 设置位置
        transform[0, 3] = center[0] - image.shape[1]/2  # 相对于图像中心的x偏移
        transform[1, 3] = center[1] - image.shape[0]/2  # 相对于图像中心的y偏移
        transform[2, 3] = -depth
        
        # 创建旋转矩阵
        rot_z = np.array([
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1]
        ])
        
        # 俯视角度（30度）
        tilt_angle = math.radians(30)
        rot_x = np.array([
            [1, 0, 0],
            [0, math.cos(tilt_angle), -math.sin(tilt_angle)],
            [0, math.sin(tilt_angle), math.cos(tilt_angle)]
        ])
        
        # 组合旋转
        final_rot = rot_z @ rot_x
        transform[:3, :3] = final_rot * scale
        
        return transform
        
    def render(self, image, detections):
        """
        渲染3D场景
        """
        # 创建场景（使用白色背景）
        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0])
        
        # 设置相机参数
        camera = pyrender.IntrinsicsCamera(
            fx=self.camera_matrix[0,0],
            fy=self.camera_matrix[1,1],
            cx=image.shape[1]/2,  # 使用图像中心
            cy=image.shape[0]/2,
            znear=1.0,
            zfar=1000.0
        )
        
        # 设置相机位置
        camera_pose = np.eye(4)
        camera_pose[1, 3] = -50  # 相机后移
        camera_pose[2, 3] = 100  # 相机抬高
        scene.add(camera, pose=camera_pose)
        
        # 添加多个光源
        # 主光源
        main_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)
        scene.add(main_light, pose=np.eye(4))
        
        # 补充光源
        side_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        side_light_pose = np.eye(4)
        side_light_pose[0, 3] = 100
        scene.add(side_light, pose=side_light_pose)
        
        # 渲染每个检测到的车辆
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
            
            try:
                mesh = pyrender.Mesh.from_trimesh(self.vehicle_mesh, smooth=True)
                scene.add(mesh, pose=transform)
            except Exception as e:
                print(f"添加模型时出错: {e}")
        
        # 渲染场景
        try:
            r = pyrender.OffscreenRenderer(image.shape[1], image.shape[0])
            color, depth = r.render(scene)
            
            # 检查渲染结果
            if color is None:
                print("渲染结果为空")
                return np.ones_like(image) * 255
            
            render_result = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            
            # 检查是否有内容
            if np.all(render_result == render_result[0,0]):
                print("渲染结果是单色的")
            
            return render_result
        except Exception as e:
            print(f"渲染时出错: {e}")
            return np.ones_like(image) * 255 