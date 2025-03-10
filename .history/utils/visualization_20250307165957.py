import numpy as np
import open3d as o3d
import cv2

class Scene3D:
    def __init__(self, vehicle_model_path):
        # 初始化可视化窗口
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        
        # 加载车辆模型
        self.vehicle_model = o3d.io.read_triangle_mesh(vehicle_model_path)
        self.vehicle_model.compute_vertex_normals()
        # 将模型设置为白色
        self.vehicle_model.paint_uniform_color([1, 1, 1])
        
        # 添加地面网格
        self.setup_ground()
        
        # 设置默认视角
        self.setup_camera()
    
    def setup_ground(self):
        # 创建地面网格
        ground = o3d.geometry.TriangleMesh.create_box(20, 20, 0.01)
        ground.translate([-10, -10, -0.005])  # 将地面置于原点下方
        ground.paint_uniform_color([0.8, 0.8, 0.8])  # 灰色
        self.vis.add_geometry(ground)
    
    def setup_camera(self):
        # 设置初始视角
        ctr = self.vis.get_view_control()
        cam = ctr.convert_to_pinhole_camera_parameters()
        # 设置相机外参（初始值，后续会根据实际情况调整）
        cam.extrinsic = np.array([
            [1, 0, 0, 0],
            [0, 0, -1, -10],
            [0, 1, 0, 10],
            [0, 0, 0, 1]
        ])
        ctr.convert_from_pinhole_camera_parameters(cam)
    
    def update_scene(self, vehicles):
        # 清除之前的车辆模型
        self.vis.clear_geometries()
        self.setup_ground()
        
        # 添加每辆车的3D模型
        for vehicle in vehicles:
            vehicle_mesh = o3d.geometry.TriangleMesh(self.vehicle_model)
            
            # 从旋转向量转换为旋转矩阵
            rotation_matrix, _ = cv2.Rodrigues(vehicle['rotation'])
            
            # 构建变换矩阵
            transform = np.eye(4)
            transform[:3, :3] = rotation_matrix
            transform[:3, 3] = vehicle['translation'].flatten()
            
            # 应用变换
            vehicle_mesh.transform(transform)
            self.vis.add_geometry(vehicle_mesh)
        
        # 更新渲染
        self.vis.poll_events()
        self.vis.update_renderer()
    
    def update_camera_params(self, camera_matrix, image_size):
        ctr = self.vis.get_view_control()
        cam = ctr.convert_to_pinhole_camera_parameters()
        
        # 设置内参
        cam.intrinsic.set_intrinsics(
            width=image_size[1],
            height=image_size[0],
            fx=camera_matrix[0, 0],
            fy=camera_matrix[1, 1],
            cx=camera_matrix[0, 2],
            cy=camera_matrix[1, 2]
        )
        
        ctr.convert_from_pinhole_camera_parameters(cam)
    
    def close(self):
        self.vis.destroy_window()
