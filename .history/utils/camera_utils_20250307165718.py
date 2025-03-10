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
    # 构建2D-3D对应点
    image_points = np.array([front_point, rear_point], dtype=np.float32)
    
    # 假设车辆中心在原点，前后点在车辆长度的一半处
    object_points = np.array([
        [vehicle_length/2, 0, 0],  # 前点
        [-vehicle_length/2, 0, 0]   # 后点
    ], dtype=np.float32)
    
    # 使用PnP求解位姿
    success, rotation_vector, translation_vector = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    return success, rotation_vector, translation_vector
