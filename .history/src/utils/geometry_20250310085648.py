import numpy as np
import numpy.typing as npt
from typing import Dict, Any

class ImageToWorldConverter:
    """图像坐标到世界坐标的转换器"""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        初始化转换器
        Args:
            config: 配置对象
        """
        self.config = config
        self._load_camera_params()
        
    def _load_camera_params(self) -> None:
        """加载相机参数"""
        try:
            calib_data = np.load(self.config['camera']['calibration_path'])
            self.camera_matrix = calib_data['camera_matrix']
            self.dist_coeffs = calib_data['dist_coeffs']
            self.camera_height = self.config['camera']['default_height']
        except Exception as e:
            raise ValueError(f"无法加载相机参数: {str(e)}")
            
    def image_to_world(self, image_point: npt.NDArray) -> npt.NDArray:
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
        
        # 计算射线与地平面的交点
        scale = self.camera_height / (y + 1e-6)  # 防止除零
        
        # 计算世界坐标
        world_x = x * scale
        world_z = scale
        world_y = 0  # 地平面高度
        
        return np.array([world_x, world_y, world_z]) 