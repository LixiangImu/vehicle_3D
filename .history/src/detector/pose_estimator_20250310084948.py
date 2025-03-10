import numpy as np
from typing import Dict, Any
import numpy.typing as npt
import logging

logger = logging.getLogger(__name__)

class VehiclePoseEstimator:
    """车辆位姿估计器类"""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        初始化位姿估计器
        Args:
            config: 配置对象
        """
        self.config = config
        self._validate_config(config)
        
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        验证配置
        Args:
            config: 配置对象
        Raises:
            ValueError: 配置不完整时
        """
        if 'model' not in config or 'default_dimensions' not in config['model']:
            raise ValueError("配置缺少车辆默认尺寸信息")
            
    def estimate_pose(self, detection: Dict[str, npt.NDArray], converter: Any) -> Dict[str, Any]:
        """
        估计车辆3D位姿
        Args:
            detection: 检测结果
            converter: 坐标转换器
        Returns:
            pose: 包含位置、角度和缩放的字典
        """
        try:
            front_3d = converter.image_to_world(detection['front_point'])
            rear_3d = converter.image_to_world(detection['rear_point'])
            
            direction = front_3d - rear_3d
            direction_xz = np.array([direction[0], direction[2]])
            angle = np.arctan2(direction_xz[0], direction_xz[1])
            
            center_3d = (front_3d + rear_3d) / 2
            vehicle_length = np.linalg.norm(front_3d - rear_3d)
            
            bbox = detection['bbox']
            image_height = bbox[3] - bbox[1]
            distance_factor = 1000 / (image_height + 1e-6)
            scale_factor = (vehicle_length / self.config['model']['default_dimensions']['length']) * (distance_factor / 100)
            
            return {
                'center': center_3d,
                'angle': angle,
                'scale': scale_factor,
                'confidence': detection.get('confidence', 1.0)
            }
            
        except Exception as e:
            logger.error(f"位姿估计失败: {str(e)}")
            return None 