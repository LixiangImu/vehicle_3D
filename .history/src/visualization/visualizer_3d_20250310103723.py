import open3d as o3d
import numpy as np
from typing import Dict, List, Any
import logging
import copy

logger = logging.getLogger(__name__)

class Visualizer3D:
    """3D可视化器类"""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        初始化3D可视化器
        Args:
            config: 配置对象
        """
        self.config = config
        try:
            self.car_model = o3d.io.read_triangle_mesh(config['model']['car_model_path'])
            if not self.car_model.has_vertices():
                raise ValueError("无法加载车辆3D模型")
        except Exception as e:
            logger.error(f"初始化3D可视化器失败: {str(e)}")
            raise
            
    def visualize(self, detections: List[Dict[str, Any]], poses: List[Dict[str, Any]]) -> None:
        """
        可视化检测结果
        Args:
            detections: 检测结果列表
            poses: 位姿估计结果列表
        """
        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window(
                width=self.config['visualization']['window']['width'],
                height=self.config['visualization']['window']['height']
            )
            
            # 设置渲染选项
            render_option = vis.get_render_option()
            render_option.background_color = np.asarray([0.8, 0.8, 0.8])
            render_option.mesh_show_back_face = True
            render_option.mesh_show_wireframe = False
            render_option.light_on = True
            render_option.point_size = 1
            render_option.line_width = 1.0
            
            self._add_coordinate_frame(vis)
            self._add_ground_grid(vis)
            self._add_vehicles(vis, poses)
            self._setup_camera(vis)
            
            vis.run()
            vis.destroy_window()
            
        except Exception as e:
            logger.error(f"可视化过程出错: {str(e)}")
            
    def _add_coordinate_frame(self, vis: o3d.visualization.Visualizer) -> None:
        """添加坐标系"""
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0, origin=[0, 0, 0])
        vis.add_geometry(coordinate_frame)
        
    def _add_ground_grid(self, vis: o3d.visualization.Visualizer) -> None:
        """添加地面网格"""
        grid_size = self.config['visualization']['grid']['size']
        grid_resolution = self.config['visualization']['grid']['resolution']
        # 实现网格创建逻辑...
        
    def _add_vehicles(self, vis: o3d.visualization.Visualizer, poses: List[Dict[str, Any]]) -> None:
        """添加车辆模型"""
        for pose in poses:
            if pose is None:
                continue
            try:
                model_copy = copy.deepcopy(self.car_model)
                R = self._get_rotation_matrix(pose['angle'])
                model_copy.rotate(R)
                model_copy.translate(pose['center'])
                model_copy.scale(pose['scale'], center=pose['center'])
                
                # 移除随机颜色设置，使用模型原始材质
                model_copy.compute_vertex_normals()
                
                vis.add_geometry(model_copy)
            except Exception as e:
                logger.error(f"添加车辆模型失败: {str(e)}")
                
    def _setup_camera(self, vis: o3d.visualization.Visualizer) -> None:
        """设置相机视角，使用外参矩阵"""
        try:
            ctr = vis.get_view_control()
            
            # 计算45度角的旋转矩阵（绕X轴）
            angle = np.deg2rad(45)
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)]
            ])
            
            # 构建相机外参矩阵
            R = Rx @ np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])
            
            # 设置相机位置（更远的距离）
            t = np.array([0, -15, -15]).reshape(3, 1)  # 同时后移和抬高相机
            
            # 从外参矩阵中提取相机位置和朝向
            camera_pos = -R.T @ t
            front = R[2, :]  # 相机的Z轴方向
            up = -R[1, :]    # 相机的Y轴方向
            
            # 设置视角
            ctr.set_front(front)
            ctr.set_lookat([0, 0, 0])
            ctr.set_up(up)
            ctr.set_zoom(0.5)  # 调整缩放以获得更好的视野
            
        except Exception as e:
            logger.error(f"设置相机参数失败: {str(e)}")
        
    def _get_rotation_matrix(self, angle: float) -> np.ndarray:
        """获取旋转矩阵"""
        return np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ]) 