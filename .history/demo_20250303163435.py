import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
import math
import torch
import copy
from ultralytics import YOLO  # 添加这行导入
import matplotlib.pyplot as plt

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
            visualized_image: YOLO检测结果的可视化图像
        """
        # 加载YOLOv8模型
        model = YOLO('pose.pt')
        
        # 预处理图像并进行检测
        results = model(image)
        
        # 获取可视化后的图像
        visualized_image = results[0].plot()
        
        # 在图像上绘制关键点
        img_with_kpts = visualized_image.copy()
        for result in results[0]:
            # 获取关键点坐标
            keypoints = result.keypoints.xy[0].cpu().numpy()
            front_point = tuple(map(int, keypoints[0]))  # 前端关键点
            rear_point = tuple(map(int, keypoints[1]))   # 后端关键点
            
            # 绘制关键点（前端点为红色，后端点为蓝色）
            cv2.circle(img_with_kpts, front_point, 5, (255, 0, 0), -1)  # 红色
            cv2.circle(img_with_kpts, rear_point, 5, (0, 0, 255), -1)   # 蓝色
            # 绘制连接线
            cv2.line(img_with_kpts, front_point, rear_point, (0, 255, 0), 2)  # 绿色线
        
        # 解析检测结果
        detections = []
        for result in results[0]:
            # 获取边界框
            bbox = result.boxes.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            
            # 获取关键点坐标
            keypoints = result.keypoints.xy[0].cpu().numpy()  # 所有关键点坐标
            front_point = keypoints[0]  # 前端关键点 [x, y]
            rear_point = keypoints[1]   # 后端关键点 [x, y]
            
            detections.append([bbox, front_point, rear_point])
        
        return detections, img_with_kpts

    def _simulate_detections(self):
        """
        生成模拟的检测数据用于测试
        """
        # 模拟数据：[[bbox], [前端关键点], [后端关键点]]
        return [
            [[100, 100, 300, 200], [150, 120], [250, 120]],
            [[400, 150, 600, 250], [450, 170], [550, 170]]
        ]

    def image_to_world_coordinates(self, image_point, bbox_y_center):
        """
        将图像坐标转换为世界坐标
        Args:
            image_point: [x, y] 图像坐标
            bbox_y_center: 边界框中心的y坐标，用于估计深度
        Returns:
            world_coords: [x, y, z] 世界坐标
        """
        # 将图像坐标转换为相机坐标系下的归一化坐标
        x = (image_point[0] - self.camera_matrix[0, 2]) / self.camera_matrix[0, 0]
        y = (image_point[1] - self.camera_matrix[1, 2]) / self.camera_matrix[1, 1]
        
        # 相机参数
        image_height = 1080  # 图像高度
        camera_height = 15.0  # 相机高度
        camera_tilt = np.radians(-30)  # 相机俯视角度
        
        # 根据边界框在图像中的位置估计深度
        # 图像底部的车更近，顶部的车更远
        normalized_y = bbox_y_center / image_height  # 归一化到[0,1]
        base_depth = 20.0  # 最近距离
        max_depth = 100.0  # 最远距离
        
        # 使用非线性映射计算深度
        depth = base_depth + (max_depth - base_depth) * (1 - normalized_y) ** 2
        
        # 计算世界坐标（考虑相机角度）
        world_x = x * depth * 1.5  # 适当放大x方向的范围
        world_z = depth
        world_y = 0  # 确保在地面上
        
        return np.array([world_x, world_y, world_z])

    def estimate_vehicle_pose(self, front_point, rear_point, bbox):
        """
        估计车辆在3D空间中的位姿
        """
        # 计算边界框中心
        bbox_y_center = (bbox[1] + bbox[3]) / 2
        
        # 转换前后点到世界坐标
        front_3d = self.image_to_world_coordinates(front_point, bbox_y_center)
        rear_3d = self.image_to_world_coordinates(rear_point, bbox_y_center)
        
        # 计算车辆朝向
        direction = front_3d - rear_3d
        angle = np.arctan2(direction[0], direction[2])
        
        # 计算车辆中心点
        center_3d = (front_3d + rear_3d) / 2
        
        # 根据YOLO检测结果估计车辆尺寸
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        image_height = 1080
        
        # 根据边界框大小和位置估计缩放因子
        # 图像底部的车应该更大，顶部的车更小
        base_scale = 0.15
        depth_scale = 1.0 + (1.0 - bbox_y_center / image_height) * 2
        size_scale = 1000 / (bbox_height + 1e-6)
        scale_factor = base_scale * size_scale / depth_scale
        
        # 根据前后点距离调整缩放
        points_distance = np.linalg.norm(front_3d - rear_3d)
        length_scale = points_distance / self.default_car_length
        scale_factor *= length_scale
        
        # 打印调试信息
        print(f"Vehicle at y={bbox_y_center:.1f}")
        print(f"Position: x={center_3d[0]:.1f}, z={center_3d[2]:.1f}")
        print(f"Scale: {scale_factor:.3f}, Angle: {np.degrees(angle):.1f}")
        print(f"Points distance: {points_distance:.1f}")
        
        return center_3d, angle, scale_factor

    def visualize_3d_scene(self, detections):
        """
        可视化3D场景
        """
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        # 添加坐标系
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=5.0, origin=[0, 0, 0])
        vis.add_geometry(coordinate_frame)
        
        # 添加更大的地平面网格
        grid_size = 200
        grid_resolution = 5
        grid_points = []
        grid_lines = []
        for i in range(-grid_size, grid_size + 1, grid_resolution):
            grid_points.extend([[i, 0, -grid_size], [i, 0, grid_size]])
            grid_points.extend([[-grid_size, 0, i], [grid_size, 0, i]])
            grid_lines.extend([[len(grid_points)-4, len(grid_points)-3],
                              [len(grid_points)-2, len(grid_points)-1]])
        grid = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(grid_points),
            lines=o3d.utility.Vector2iVector(grid_lines)
        )
        grid.paint_uniform_color([0.8, 0.8, 0.8])
        vis.add_geometry(grid)
        
        # 处理每个检测到的车辆
        for detection in detections:
            bbox, front_point, rear_point = detection
            
            try:
                # 获取车辆位姿
                center, angle, scale = self.estimate_vehicle_pose(
                    front_point, rear_point, bbox)
                
                # 复制并变换3D模型
                car_model_transformed = copy.deepcopy(self.car_model)
                
                # 应用变换
                # 1. 缩放
                car_model_transformed.scale(scale, center=car_model_transformed.get_center())
                
                # 2. 旋转（绕Y轴）
                R = car_model_transformed.get_rotation_matrix_from_xyz((0, angle, 0))
                car_model_transformed.rotate(R, center=car_model_transformed.get_center())
                
                # 3. 平移到目标位置
                car_model_transformed.translate(center)
                
                # 为每个车辆设置随机颜色
                car_model_transformed.paint_uniform_color([
                    np.random.uniform(0.3, 0.9),
                    np.random.uniform(0.3, 0.9),
                    np.random.uniform(0.3, 0.9)
                ])
                
                vis.add_geometry(car_model_transformed)
                
            except Exception as e:
                print(f"处理车辆时出错: {e}")
                continue
        
        # 设置渲染选项
        render_option = vis.get_render_option()
        render_option.background_color = np.asarray([0.9, 0.9, 0.9])
        render_option.point_size = 1
        render_option.line_width = 1.0
        
        # 设置相机视角
        ctr = vis.get_view_control()
        ctr.set_zoom(0.3)
        ctr.set_front([0, -1, -0.3])    # 从高处向下看
        ctr.set_lookat([0, 0, 20])      # 看向远处
        ctr.set_up([0, 1, 0])           # 保持y轴向上
        
        vis.run()
        vis.destroy_window()
        
        return None

def main():
    # 初始化参数
    camera_calib_path = "camera_calibration.npz"
    model_path = r"D:\Desktop\yitaixx\3D_whiteModel\3Dmodel\car.obj"
    image_path = r"D:\Desktop\yitaixx\3D_whiteModel\data\car\car2.jpg"
    
    # 创建检测器实例
    detector = VehicleDetector3D(camera_calib_path, model_path)
    
    # 读取图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB
    
    # 检测车辆并获取可视化结果
    detections, yolo_vis_image = detector.detect_vehicles(image)
    
    # 创建图像窗口
    plt.figure(figsize=(12, 6))
    plt.subplot(121)  # 1行2列的第1个
    plt.title('YOLO Detection with Keypoints')
    plt.imshow(yolo_vis_image)
    plt.axis('off')
    
    # 更新显示（但不阻塞）
    plt.draw()
    plt.pause(0.1)  # 短暂暂停以确保图像显示
    
    # 显示3D渲染结果（交互式窗口）
    detector.visualize_3d_scene(detections)
    
    # 保持matplotlib窗口打开
    plt.show()

if __name__ == "__main__":
    main()