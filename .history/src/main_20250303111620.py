import cv2
import numpy as np
from vehicle_detector import VehicleDetector
from scene_renderer import SceneRenderer

def resize_image(image, target_size=1280):
    """
    等比例缩放图像，保持长边为target_size
    """
    height, width = image.shape[:2]
    scale = target_size / max(height, width)
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # 创建方形画布
    square_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    # 将调整后的图像放在中心
    y_offset = (target_size - new_height) // 2
    x_offset = (target_size - new_width) // 2
    square_img[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    return square_img, scale, (x_offset, y_offset)

def draw_detections(image, detections):
    """
    在图像上绘制YOLO检测结果
    """
    result = image.copy()
    for det in detections:
        bbox = det['bbox'].astype(int)
        front_point = det['front_point'].astype(int)
        rear_point = det['rear_point'].astype(int)
        
        # 绘制边界框
        cv2.rectangle(result, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        # 绘制前后点
        cv2.circle(result, tuple(front_point), 5, (0, 0, 255), -1)  # 前点红色
        cv2.circle(result, tuple(rear_point), 5, (255, 0, 0), -1)   # 后点蓝色
        
        # 绘制方向线
        cv2.line(result, tuple(rear_point), tuple(front_point), (0, 255, 255), 2)
        
    return result

def create_side_by_side_view(img1, img2, target_size=1280):
    """
    创建并排显示的视图
    """
    # 创建黑色背景
    combined = np.zeros((target_size, target_size * 2, 3), dtype=np.uint8)
    
    # 将两张图片并排放置
    combined[:, :target_size] = img1
    combined[:, target_size:] = img2
    
    # 添加分割线
    cv2.line(combined, (target_size, 0), (target_size, target_size), (255, 255, 255), 2)
    
    return combined

def main():
    # 初始化检测器
    detector = VehicleDetector(
        weights_path='pose.pt'
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
    
    # 缩放图像到1280*1280
    resized_image, scale, offsets = resize_image(image)
    
    # 检测车辆
    detections = detector.detect(resized_image)
    
    # 在图像上绘制YOLO检测结果
    detection_vis = draw_detections(resized_image, detections)
    
    # 渲染3D场景
    render_result = renderer.render(resized_image, detections)
    
    # 创建并排显示的视图
    combined_view = create_side_by_side_view(detection_vis, render_result)
    
    # 创建窗口并显示结果
    cv2.namedWindow("Vehicle Detection and 3D Rendering", cv2.WINDOW_NORMAL)
    cv2.imshow("Vehicle Detection and 3D Rendering", combined_view)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 保存结果
    cv2.imwrite("output_combined.jpg", combined_view)

if __name__ == "__main__":
    main() 