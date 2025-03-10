import torch
from ultralytics import YOLO

# 加载 YOLOv8 关键点检测模型
model = YOLO("pose.pt")  # 或者你的自训练模型

# 预测图像
results = model(r"data\car\car2.jpg")

# 遍历每个检测结果
for result in results:
    boxes = result.boxes.xywh  # (x_center, y_center, width, height)
    confidences = result.boxes.conf  # 目标置信度
    keypoints = result.keypoints.xy  # (num_detections, num_keypoints, 2)  仅 (x, y)

    print("检测框坐标（中心点 + 宽高）:", boxes)
    print("目标置信度:", confidences)
    print("关键点坐标:", keypoints)
