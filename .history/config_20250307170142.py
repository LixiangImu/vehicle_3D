import numpy as np

# 相机参数
CAMERA_MATRIX = np.array([
    [4.07252926e+03, 0.00000000e+00, 2.03582422e+03],
    [0.00000000e+00, 4.05829021e+03, 2.84855995e+03],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

DIST_COEFFS = np.array([
    [3.48338040e-01, -3.11647655e+00, 1.94125582e-03, -1.83458110e-02, 5.18275125e+00]
])

# 图像参数
IMAGE_WIDTH = 4032
IMAGE_HEIGHT = 3024
DOWNSAMPLE_FACTOR = 2  # 降采样因子，设为2表示将图像缩小一半

# 3D模型参数
VEHICLE_LENGTH = 4.5  # 标准车长（米）
VEHICLE_WIDTH = 1.8   # 标准车宽（米）
VEHICLE_HEIGHT = 1.5  # 标准车高（米）

# 模型路径
YOLO_MODEL_PATH = "models/pose.pt"
VEHICLE_MODEL_PATH = "models/car.obj"

# 可视化参数
VISUALIZATION_WINDOW_NAME = "3D Vehicle Visualization"
