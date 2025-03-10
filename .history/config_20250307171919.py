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
DOWNSAMPLE_FACTOR = 8  # 增大降采样因子以匹配YOLO输入

# 3D模型参数
VEHICLE_LENGTH = 4.5  # 标准车长（米）
VEHICLE_WIDTH = 1.8   # 标准车宽（米）
VEHICLE_HEIGHT = 1.5  # 标准车高（米）

# 模型路径
YOLO_MODEL_PATH = "models/pose.pt"
VEHICLE_MODEL_PATH = "models/car.obj"

# 可视化参数
VISUALIZATION_WINDOW_WIDTH = 1280   # 添加窗口尺寸配置
VISUALIZATION_WINDOW_HEIGHT = 720
VISUALIZATION_WINDOW_NAME = "3D Vehicle Visualization"

# 3D场景参数
GROUND_SIZE = 50.0  # 增大地面尺寸
GROUND_GRID_SIZE = 1.0  # 网格大小
CAMERA_HEIGHT = 15.0  # 预设相机高度
SCENE_SCALE = 5.0  # 场景整体缩放系数
VEHICLE_SPACING = 2.0  # 车辆之间的最小间距
