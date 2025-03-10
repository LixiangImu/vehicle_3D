import numpy as np

class VehiclePoseEstimator:
    def __init__(self, config):
        self.config = config
        
    def estimate_pose(self, detection, converter):
        """
        估计车辆3D位姿
        Args:
            detection: 检测结果
            converter: 坐标转换器
        Returns:
            pose: 包含位置、角度和缩放的字典
        """
        front_3d = converter.image_to_world(detection['front_point'])
        rear_3d = converter.image_to_world(detection['rear_point'])
        
        direction = front_3d - rear_3d
        direction_xz = np.array([direction[0], direction[2]])
        angle = np.arctan2(direction_xz[0], direction_xz[1])
        
        center_3d = (front_3d + rear_3d) / 2
        vehicle_length = np.linalg.norm(front_3d - rear_3d)
        
        bbox = detection['bbox']
        image_height = bbox[3] - bbox[1]
        distance_factor = 1000 / (image_height + 1e-6)
        scale_factor = (vehicle_length / self.config['model']['default_dimensions']['length']) * (distance_factor / 100)
        
        return {
            'center': center_3d,
            'angle': angle,
            'scale': scale_factor
        } 