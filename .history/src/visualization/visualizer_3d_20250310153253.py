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
            # 创建窗口
            vis = o3d.visualization.Visualizer()
            window_width = self.config['visualization']['window']['width']
            window_height = self.config['visualization']['window']['height']
            vis.create_window(width=window_width, height=window_height)
            
            # 添加几何体
            self._add_coordinate_frame(vis)
            self._add_ground_grid(vis)
            self._add_vehicles(vis, poses)
            
            # 设置渲染选项
            render_option = vis.get_render_option()
            render_option.background_color = np.asarray([0.8, 0.8, 0.8])
            render_option.mesh_show_back_face = True
            render_option.mesh_show_wireframe = False
            render_option.light_on = True
            render_option.point_size = 1
            render_option.line_width = 1.0
            
            # 等待窗口创建完成
            vis.poll_events()
            vis.update_renderer()
            
            # 创建相机参数
            ctr = vis.get_view_control()
            params = o3d.camera.PinholeCameraParameters()
            
            # 确保内参矩阵与窗口尺寸匹配
            params.intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=window_width,
                height=window_height,
                fx=window_width,
                fy=window_height,
                cx=window_width / 2,
                cy=window_height / 2
            )
            
            # 设置外参矩阵
            params.extrinsic = np.array([
                [-0.99999, -0.0006611, -0.005333, 3.012],
                [0.0035022, -0.83287, -0.55345, -8.7022],
                [-0.0040759, -0.55346, 0.83286, 64.412],
                [0, 0, 0, 1]
            ])
            
            # 应用相机参数并等待更新
            ctr.convert_from_pinhole_camera_parameters(params)
            vis.poll_events()
            vis.update_renderer()
            
            # 运行可视化窗口
            vis.run()
            
            # 获取调整后的相机参数（用于调试）
            final_params = ctr.convert_to_pinhole_camera_parameters()
            print("\n当前相机参数:")
            print(f"外参矩阵:\n{final_params.extrinsic}")
            print(f"相机位置: {-final_params.extrinsic[:3, :3].T @ final_params.extrinsic[:3, 3]}")
            print(f"相机朝向: {final_params.extrinsic[:3, :3][:, 2]}")
            print(f"相机上方向: {final_params.extrinsic[:3, :3][:, 1]}")
            
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
                
    def _get_rotation_matrix(self, angle: float) -> np.ndarray:
        """获取旋转矩阵"""
        return np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ]) 