import numpy as np
import open3d as o3d
import cv2
import config

class Scene3D:
    def __init__(self, vehicle_model_path):
        # 初始化可视化窗口
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            width=config.VISUALIZATION_WINDOW_WIDTH,
            height=config.VISUALIZATION_WINDOW_HEIGHT
        )
        
        # 加载车辆模型
        self.vehicle_model = o3d.io.read_triangle_mesh(vehicle_model_path)
        self.vehicle_model.compute_vertex_normals()
        self.vehicle_model.paint_uniform_color([0.8, 0.8, 0.8])  # 浅灰色
        
        # 添加地面网格
        self.setup_ground()
        
        # 设置默认视角
        self.setup_camera()
        
        # 设置渲染选项
        self.setup_render_options()
    
    def setup_ground(self):
        # 创建更大的地面网格
        ground = o3d.geometry.TriangleMesh.create_box(
            config.GROUND_SIZE,
            config.GROUND_SIZE,
            0.01
        )
        ground.translate([
            -config.GROUND_SIZE/2,
            -config.GROUND_SIZE/2,
            -0.005
        ])
        ground.paint_uniform_color([0.9, 0.9, 0.9])  # 更浅的灰色
        self.vis.add_geometry(ground)
        
        # 添加网格线
        self.add_grid()
    
    def add_grid(self):
        # 添加网格线以提供参考
        for i in range(int(-config.GROUND_SIZE/2), int(config.GROUND_SIZE/2) + 1, int(config.GROUND_GRID_SIZE)):
            # 创建X方向的线
            line_x = o3d.geometry.LineSet()
            line_x.points = o3d.utility.Vector3dVector([
                [i, -config.GROUND_SIZE/2, 0],
                [i, config.GROUND_SIZE/2, 0]
            ])
            line_x.lines = o3d.utility.Vector2iVector([[0, 1]])
            line_x.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5]])
            self.vis.add_geometry(line_x)
            
            # 创建Y方向的线
            line_y = o3d.geometry.LineSet()
            line_y.points = o3d.utility.Vector3dVector([
                [-config.GROUND_SIZE/2, i, 0],
                [config.GROUND_SIZE/2, i, 0]
            ])
            line_y.lines = o3d.utility.Vector2iVector([[0, 1]])
            line_y.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5]])
            self.vis.add_geometry(line_y)
    
    def setup_render_options(self):
        render_option = self.vis.get_render_option()
        render_option.background_color = np.array([1, 1, 1])  # 白色背景
        render_option.point_size = 5.0
        render_option.line_width = 1.0
        render_option.light_on = True
    
    def setup_camera(self):
        ctr = self.vis.get_view_control()
        cam = ctr.convert_to_pinhole_camera_parameters()
        # 设置一个俯视角度的默认视角
        cam.extrinsic = np.array([
            [1, 0, 0, 0],
            [0, 0.7071, -0.7071, -config.CAMERA_HEIGHT],
            [0, 0.7071, 0.7071, config.CAMERA_HEIGHT],
            [0, 0, 0, 1]
        ])
        ctr.convert_from_pinhole_camera_parameters(cam)
    
    def update_scene(self, vehicles):
        # 清除之前的车辆模型
        self.vis.clear_geometries()
        self.setup_ground()  # 重新添加地面
        
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
