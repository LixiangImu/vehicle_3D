import cv2
import yaml
from pathlib import Path
from src.detector import VehicleDetector
from src.visualization import Visualizer2D, Visualizer3D

def main():
    # 加载配置
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 初始化检测器和可视化器
    detector = VehicleDetector(config)
    vis_2d = Visualizer2D(config)
    vis_3d = Visualizer3D(config)
    
    # 读取测试图像
    image_path = "data/images/test.jpg"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 执行检测
    detections = detector.detect(image)
    
    # 可视化结果
    vis_2d.show_detections(image, detections)
    vis_3d.visualize(detections)

if __name__ == "__main__":
    main() 