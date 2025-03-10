import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
import math
import torch
import copy
from ultralytics import YOLO
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

        # 初始化相机外参
        self.R = np.eye(3)  # 旋转矩阵
        self.t = np.zeros((3, 1))  # 平移向量

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

    def convert_camera_parameters(self):
        """
        将OpenCV相机参数转换为Open3D格式
        """
        # 创建Open3D相机参数对象
        param = o3d.camera.PinholeCameraParameters()
        
        # 设置内参矩阵
        param.intrinsic = o3d.camera.PinholeCameraIntrinsic()
        param.intrinsic.intrinsic_matrix = self.camera_matrix
        
        # 计算外参矩阵的转换
        # 1. 创建坐标系转换矩阵（从OpenCV到Open3D）
        cv_to_o3d = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],  # Y轴翻转
            [0, 0, -1, 0],  # Z轴翻转
            [0, 0, 0, 1]
        ])
        
        # 2. 构建OpenCV格式的4x4变换矩阵
        cv_extrinsic = np.eye(4)
        cv_extrinsic[:3, :3] = self.R
        cv_extrinsic[:3, 3] = self.t.flatten()
        
        # 3. 转换为Open3D格式
        o3d_extrinsic = cv_to_o3d @ cv_extrinsic @ np.linalg.inv(cv_to_o3d)
        
        # 4. 设置外参
        param.extrinsic = o3d_extrinsic
        
        return param

    def estimate_vehicle_pose(self, front_point, rear_point, bbox):
        """
        估计车辆在3D空间中的位姿
        """
        # 转换前后点到世界坐标
        front_3d = self.image_to_world_coordinates(front_point)
        rear_3d = self.image_to_world_coordinates(rear_point)
        
        # 计算车辆朝向
        direction = front_3d - rear_3d
        direction_xz = np.array([direction[0], direction[2]])
        angle = np.arctan2(direction_xz[0], direction_xz[1])
        
        # 计算车辆中心点
        center_3d = (front_3d + rear_3d) / 2
        
        # 估计车辆尺寸
        vehicle_length = np.linalg.norm(front_3d - rear_3d)
        scale_factor = vehicle_length / self.default_car_length
        
        # 根据图像中的位置调整缩放
        image_height = bbox[3] - bbox[1]
        distance_factor = 1000 / (image_height + 1e-6)
        scale_factor *= distance_factor / 100
        
        # 更新相机外参（这里需要根据实际情况实现）
        # 示例：使用简单的视角估计
        look_at = center_3d
        up = np.array([0, 1, 0])
        camera_pos = center_3d + np.array([0, 10, -20])  # 相机位置在车辆上方后方
        
        # 计算旋转矩阵
        z_axis = look_at - camera_pos
        z_axis = z_axis / np.linalg.norm(z_axis)
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        
        self.R = np.column_stack((x_axis, y_axis, z_axis))
        self.t = -self.R @ camera_pos.reshape(3, 1)
        
        return center_3d, angle, scale_factor

    def image_to_world_coordinates(self, image_point):
        """
        将图像坐标转换为世界坐标
        Args:
            image_point: [x, y] 图像坐标
        Returns:
            world_coords: [x, y, z] 世界坐标
        """
        # 将图像坐标转换为相机坐标系下的归一化坐标
        x = (image_point[0] - self.camera_matrix[0, 2]) / self.camera_matrix[0, 0]
        y = (image_point[1] - self.camera_matrix[1, 2]) / self.camera_matrix[1, 1]
        
        # 假设相机高度为10米，朝向地面
        camera_height = 10.0
        
        # 计算射线与地平面的交点
        # 假设地平面为 z = 0
        scale = camera_height / (y + 1e-6)  # 防止除零
        
        # 计算世界坐标
        world_x = x * scale
        world_z = scale
        world_y = 0  # 地平面高度
        
        return np.array([world_x, world_y, world_z])

    def visualize_3d_scene(self, detections):
        """
        可视化3D场景并返回渲染图像
        """
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        # 添加坐标系
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=5.0, origin=[0, 0, 0])
        vis.add_geometry(coordinate_frame)
        
        # 添加地平面网格
        grid_size = 50
        grid_resolution = 1
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
        grid.paint_uniform_color([0.5, 0.5, 0.5])
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
                
                # 首先将模型移动到原点
                car_model_transformed.translate(-car_model_transformed.get_center())
                
                # 1. 缩放
                car_model_transformed.scale(scale, center=[0, 0, 0])
                
                # 2. 旋转（绕Y轴）
                R = car_model_transformed.get_rotation_matrix_from_xyz((0, angle, 0))
                car_model_transformed.rotate(R, center=[0, 0, 0])
                
                # 3. 平移到目标位置
                target_position = center.copy()
                target_position[1] = self.default_car_height * 0.5  # 将模型抬高到车身高度的一半
                car_model_transformed.translate(target_position)
                
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
        render_option.background_color = np.asarray([0.8, 0.8, 0.8])
        render_option.point_size = 1
        render_option.line_width = 1.0
        
        # 设置相机视角
        ctr = vis.get_view_control()
        camera_params = self.convert_camera_parameters()
        ctr.convert_from_pinhole_camera_parameters(camera_params)
        
        # 使用预设的相机参数（从之前保存的最佳视角）
        front = [-0.99652543453698439, -0.014639442984944209, 0.081992347386600481]
        lookat = [9.44947193726472, -9.4050516657642476, 95.881614650936953]
        up = [0.080792649592174623, -0.40911730316633693, 0.90889800309043456]
        zoom = 0.7
        
        ctr.set_front(front)
        ctr.set_lookat(lookat)
        ctr.set_up(up)
        ctr.set_zoom(zoom)
        
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