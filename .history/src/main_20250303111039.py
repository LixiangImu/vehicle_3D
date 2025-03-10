import cv2
import numpy as np
from vehicle_detector import VehicleDetector
from scene_renderer import SceneRenderer

def resize_image(image, target_width=1920):
    """
    等比例缩放图像
    Args:
        image: 输入图像
        target_width: 目标宽度
    Returns:
        缩放后的图像
    """
    height, width = image.shape[:2]
    scale = target_width / width
    target_height = int(height * scale)
    
    # 确保高度不超过1200像素
    if target_height > 1200:
        scale = 1200 / height
        target_width = int(width * scale)
        target_height = 1200
    
    resized = cv2.resize(image, (target_width, target_height), 
                        interpolation=cv2.INTER_AREA)
    return resized, scale

def main():
    # 初始化检测器
    detector = VehicleDetector(
        weights_path=r'D:\Desktop\yitaixx\3D_whiteModel\pose.pt'
    )
    
    # 初始化渲染器
    renderer = SceneRenderer(
        camera_calib_file=r'D:\Desktop\yitaixx\3D_whiteModel\camera_calibration.npz',
        model_path=r"D:\Desktop\yitaixx\3D_whiteModel\3Dmodel\car.obj"
    )
    
    # 读取图像
    image_path = r"D:\Desktop\yitaixx\3D_whiteModel\data\car\car1.jpg"
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 缩放图像
    resized_image, scale = resize_image(image)
    
    # 检测车辆
    detections = detector.detect(resized_image)
    
    # 渲染3D场景
    result = renderer.render(resized_image, detections)
    
    # 创建窗口并设置为可调整大小
    cv2.namedWindow("3D Vehicle Visualization", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("3D Vehicle Visualization", result.shape[1], result.shape[0])
    
    # 显示结果
    cv2.imshow("3D Vehicle Visualization", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 保存结果
    cv2.imwrite("output_result.jpg", result)

if __name__ == "__main__":
    main() 