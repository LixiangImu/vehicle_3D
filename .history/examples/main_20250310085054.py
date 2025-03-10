import cv2
import yaml
import logging
from pathlib import Path
from src.detector import VehicleDetector
from src.visualization import Visualizer2D, Visualizer3D

def setup_logging(config):
    """设置日志配置"""
    logging.basicConfig(
        level=config['logging']['level'],
        format=config['logging']['format']
    )

def main():
    try:
        # 加载配置
        config_path = Path('configs/config.yaml')
        if not config_path.exists():
            raise FileNotFoundError("配置文件不存在")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # 设置日志
        setup_logging(config)
        logger = logging.getLogger(__name__)
        
        # 初始化检测器和可视化器
        detector = VehicleDetector(config)
        vis_2d = Visualizer2D(config)
        vis_3d = Visualizer3D(config)
        
        # 读取测试图像
        image_path = Path("data\car\car2.jpg")
        if not image_path.exists():
            raise FileNotFoundError("测试图像不存在")
            
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError("无法读取图像")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 执行检测
        detections = detector.detect(image)
        if not detections:
            logger.warning("未检测到任何车辆")
            return
            
        # 估计位姿
        poses = [detector.pose_estimator.estimate_pose(det, detector.converter) 
                for det in detections]
        
        # 可视化结果
        vis_2d.show_detections(image, detections)
        vis_3d.visualize(detections, poses)
        
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        raise

if __name__ == "__main__":
    main() 