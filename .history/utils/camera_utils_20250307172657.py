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
    """估计车辆的3D姿态，考虑模型坐标系：-Y为前，+X为左"""
    try:
        # 计算车辆方向向量
        direction = front_point - rear_point
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-6:
            return False, None, None
        
        # 归一化方向向量
        direction = direction / direction_norm
        
        # 计算垂直于方向的向量（车辆宽度方向）
        normal = np.array([-direction[1], direction[0]])
        
        # 使用实际车辆尺寸
        half_length = vehicle_length / 2
        half_width = vehicle_length * 0.4  # 假设宽度是长度的0.4倍
        
        # 车辆中心点
        center = (front_point + rear_point) / 2
        
        # 生成3D模型点，适应模型坐标系（-Y为前，+X为左）
        object_points = np.array([
            [ half_width, -half_length, 0],  # 左前
            [-half_width, -half_length, 0],  # 右前
            [-half_width,  half_length, 0],  # 右后
            [ half_width,  half_length, 0],  # 左后
            [0,          -half_length, 0],   # 前中
            [0,           half_length, 0],   # 后中
        ], dtype=np.float32)
        
        # 生成对应的图像点
        image_points = np.array([
            front_point + normal * half_width,    # 左前
            front_point - normal * half_width,    # 右前
            rear_point - normal * half_width,     # 右后
            rear_point + normal * half_width,     # 左后
            front_point,                          # 前中
            rear_point,                           # 后中
        ], dtype=np.float32)
        
        # 使用SOLVEPNP_ITERATIVE方法求解PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success:
            # 调整平移向量的尺度
            translation_vector[2] = abs(translation_vector[2])  # 确保Z值为正
            
            # 根据图像中的车辆大小调整尺度
            scale = vehicle_length / (2 * np.linalg.norm(front_point - rear_point))
            translation_vector *= scale
            
            # 添加额外的90度旋转，使模型方向与检测到的方向对齐
            R_adjust, _ = cv2.Rodrigues(np.array([0, 0, np.pi/2]))
            R_current, _ = cv2.Rodrigues(rotation_vector)
            R_final = R_current @ R_adjust
            rotation_vector, _ = cv2.Rodrigues(R_final)
            
            return True, rotation_vector, translation_vector
            
    except Exception as e:
        print(f"Position estimation error: {e}")
        return False, None, None
    
    return False, None, None
