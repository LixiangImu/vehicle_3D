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
            
            # 设置并锁定相机视角
            view_control = vis.get_view_control()
            self._setup_camera(vis)
            view_control.change_field_of_view(step=0)  # 锁定视角
            
            # 运行可视化窗口
            vis.run()
            
            # 获取调整后的相机参数
            view_control = vis.get_view_control()
            params = view_control.convert_to_pinhole_camera_parameters()
            
            # 打印相机参数
            print("\n调整后的相机参数:")
            print(f"外参矩阵:\n{params.extrinsic}")
            print(f"相机位置: {-params.extrinsic[:3, :3].T @ params.extrinsic[:3, 3]}")
            print(f"相机朝向: {params.extrinsic[:3, :3][:, 2]}")
            print(f"相机上方向: {params.extrinsic[:3, :3][:, 1]}")
            
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
            
            # 调整后的外参矩阵，使车辆在画面下方且正确朝向
            extrinsic = np.array([
                [    -0.9991    0.032834   -0.026761  2.2204e-16],
 [  -0.020492    -0.92759    -0.37304 -3.5527e-15],
 [  -0.037072    -0.37216     0.92743      73.654],
                [0, 0, 0, 1]
            ])
            
            # 从外参矩阵中提取相机参数
            R = extrinsic[:3, :3]
            t = extrinsic[:3, 3]
            
            # 计算相机位置和方向
            camera_pos = -R.T @ t
            front = R[:, 2]    # 相机的Z轴方向（第三列）
            up = -R[:, 1]      # 相机的Y轴方向（第二列）取负
            
            # 设置视角
            ctr.set_front(front)
            ctr.set_lookat([0, 0, 0])
            ctr.set_up(up)
            ctr.set_zoom(0.7)
            
            # 锁定视角
            params = o3d.camera.PinholeCameraParameters()
            params.extrinsic = extrinsic
            ctr.convert_from_pinhole_camera_parameters(params)
            
        except Exception as e:
            logger.error(f"设置相机参数失败: {str(e)}")
        
    def _get_rotation_matrix(self, angle: float) -> np.ndarray:
        """获取旋转矩阵"""
        return np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ]) 