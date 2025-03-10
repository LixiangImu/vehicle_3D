import numpy as np
import cv2

def resize_camera_matrix(camera_matrix, scale_factor):
    """根据图像缩放调整相机内参矩阵"""
    camera_matrix_scaled = camera_matrix.copy()
    camera_matrix_scaled[0, 0] = camera_matrix_scaled[0, 0] / scale_factor
    camera_matrix_scaled[1, 1] = camera_matrix_scaled[1, 1] / scale_factor
    camera_matrix_scaled[0, 2] = camera_matrix_scaled[0, 2] / scale_factor
    camera_matrix_scaled[1, 2] = camera_matrix_scaled[1, 2] / scale_factor
    return camera_matrix_scaled

def estimate_vehicle_pose(front_point, rear_point, camera_matrix, dist_coeffs, vehicle_length):
    """估计车辆的3D姿态"""
    # 从前后点生成4个角点
    center_point = (front_point + rear_point) / 2
    direction = front_point - rear_point
    direction_norm = np.linalg.norm(direction)
    if direction_norm == 0:
        return False, None, None
    
    direction = direction / direction_norm
    normal = np.array([-direction[1], direction[0]])  # 垂直于方向的向量
    
    # 假设车宽为车长的0.4
    half_width = vehicle_length * 0.4 / 2
    
    # 生成4个角点（按顺序：左前、右前、右后、左后）
    image_points = np.array([
        front_point + normal * half_width,    # 左前
        front_point - normal * half_width,    # 右前
        rear_point - normal * half_width,     # 右后
        rear_point + normal * half_width      # 左后
    ], dtype=np.float32)
    
    # 定义3D模型点（假设车辆中心在原点）
    object_points = np.array([
        [vehicle_length/2,  vehicle_length*0.4/2, 0],  # 左前
        [vehicle_length/2, -vehicle_length*0.4/2, 0],  # 右前
        [-vehicle_length/2, -vehicle_length*0.4/2, 0], # 右后
        [-vehicle_length/2,  vehicle_length*0.4/2, 0]  # 左后
    ], dtype=np.float32)
    
    # 使用PnP求解位姿
    try:
        success, rotation_vector, translation_vector = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        return success, rotation_vector, translation_vector
    except cv2.error:
        return False, None, None
