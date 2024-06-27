from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model(source=0, show=True, conf=0.8, save=True)