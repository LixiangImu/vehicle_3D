import cv2
import numpy as np
from vehicle_detector import VehicleDetector
from scene_renderer import SceneRenderer

def main():
    # 初始化检测器
    detector = VehicleDetector(
        weights_path=r'D:\Desktop\yitaixx\3D_whiteModel\pose.pt',
        config_path='path/to/your/yolo/config.yaml'
    )
    
    # 初始化渲染器
    renderer = SceneRenderer(
        camera_calib_file='camera_calibration.npz',
        model_path=r"D:\Desktop\yitaixx\3D_whiteModel\3Dmodel\car.obj"
    )
    
    # 读取图像
    image_path = "input_image.jpg"
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 检测车辆
    detections = detector.detect(image)
    
    # 渲染3D场景
    result = renderer.render(image, detections)
    
    # 显示结果
    cv2.imshow("3D Vehicle Visualization", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 保存结果
    cv2.imwrite("output_result.jpg", result)

if __name__ == "__main__":
    main() 