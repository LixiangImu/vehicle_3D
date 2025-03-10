import logging
from typing import List, Dict, Any, Optional
import numpy as np
import numpy.typing as npt
from ultralytics import YOLO
from ..utils.geometry import ImageToWorldConverter
from .pose_estimator import VehiclePoseEstimator

logger = logging.getLogger(__name__)

class VehicleDetector:
    """车辆检测器类"""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        初始化车辆检测器
        Args:
            config: 配置对象
        Raises:
            ValueError: 配置验证失败时
        """
        self._validate_config(config)
        self.config = config
        
        try:
            self.model = YOLO(config['model']['yolo_path'])
            self.converter = ImageToWorldConverter(config)
            self.pose_estimator = VehiclePoseEstimator(config)
        except Exception as e:
            logger.error(f"初始化失败: {str(e)}")
            raise

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        验证配置是否完整
        Args:
            config: 配置对象
        Raises:
            ValueError: 配置不完整时
        """
        required_keys = ['model', 'camera', 'visualization']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"配置缺少必要的键: {key}")
        
        if 'yolo_path' not in config['model']:
            raise ValueError("配置缺少 model.yolo_path")

    def _validate_image(self, image: npt.NDArray) -> None:
        """
        验证输入图像
        Args:
            image: 输入图像
        Raises:
            ValueError: 图像格式不正确时
        """
        if image is None:
            raise ValueError("输入图像不能为空")
        if not isinstance(image, np.ndarray):
            raise TypeError("输入图像必须是numpy数组")
        if len(image.shape) != 3:
            raise ValueError("输入图像必须是3通道图像")

    def detect(self, image: npt.NDArray) -> List[Dict[str, npt.NDArray]]:
        """
        检测图像中的车辆
        Args:
            image: 输入图像
        Returns:
            detections: 检测结果列表
        """
        try:
            self._validate_image(image)
            results = self.model(image)
            detections = []
            
            if not results or len(results) == 0:
                logger.warning("未检测到任何目标")
                return []
                
            for result in results[0]:
                try:
                    if not result.boxes or not result.keypoints:
                        continue
                        
                    bbox = result.boxes.xyxy[0].cpu().numpy()
                    keypoints = result.keypoints.xy[0].cpu().numpy()
                    
                    if len(keypoints) < 2:
                        logger.warning("关键点数量不足")
                        continue
                        
                    front_point = keypoints[0]
                    rear_point = keypoints[1]
                    
                    detection = {
                        'bbox': bbox,
                        'front_point': front_point,
                        'rear_point': rear_point,
                        'confidence': float(result.boxes.conf[0])
                    }
                    
                    detections.append(detection)
                    
                except Exception as e:
                    logger.error(f"处理单个检测结果时出错: {str(e)}")
                    continue
                    
            return detections
            
        except Exception as e:
            logger.error(f"车辆检测过程出错: {str(e)}")
            return []

    def __del__(self) -> None:
        """释放资源"""
        if hasattr(self, 'model'):
            del self.model 