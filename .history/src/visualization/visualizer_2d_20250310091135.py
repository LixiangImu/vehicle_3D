import cv2
import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class Visualizer2D:
    """2D可视化器类"""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        初始化2D可视化器
        Args:
            config: 配置对象
        """
        self.config = config
        
    def show_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> None:
        """
        显示检测结果
        Args:
            image: 输入图像
            detections: 检测结果列表
        """
        try:
            vis_image = image.copy()
            
            for detection in detections:
                # 绘制边界框
                bbox = detection['bbox'].astype(int)
                cv2.rectangle(vis_image, 
                            (bbox[0], bbox[1]), 
                            (bbox[2], bbox[3]), 
                            (0, 255, 0), 2)
                
                # 绘制关键点
                front_point = detection['front_point'].astype(int)
                rear_point = detection['rear_point'].astype(int)
                
                # 前点为红色，后点为蓝色
                cv2.circle(vis_image, tuple(front_point), 5, (255, 0, 0), -1)
                cv2.circle(vis_image, tuple(rear_point), 5, (0, 0, 255), -1)
                
                # 绘制连接线
                cv2.line(vis_image, tuple(front_point), tuple(rear_point), (0, 255, 0), 2)
                
                # 显示置信度
                if 'confidence' in detection:
                    conf_text = f"{detection['confidence']:.2f}"
                    cv2.putText(vis_image, conf_text, 
                              (bbox[0], bbox[1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 缩放图像到指定大小
            display_width = self.config['visualization'].get('display_width', 800)
            display_height = self.config['visualization'].get('display_height', 600)
            vis_image = cv2.resize(vis_image, (display_width, display_height))
            
            # 显示图像
            cv2.namedWindow('Vehicle Detection', cv2.WINDOW_NORMAL)
            cv2.imshow('Vehicle Detection', cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)  # 使用较短的等待时间，避免阻塞
            
        except Exception as e:
            logger.error(f"2D可视化失败: {str(e)}") 