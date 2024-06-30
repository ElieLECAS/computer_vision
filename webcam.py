from ultralytics import YOLO

model = YOLO('yolov8_fall_detection.pt')

source = 0

results = model(source, show=True, conf=0.5, save=True)


# from roboflow import Roboflow
# rf = Roboflow(api_key="SX9eIjqIeQ6q27TDMHcc")
# project = rf.workspace("roboflow-universe-projects").project("fall-detection-ca3o8")
# version = project.version(4)
# dataset = version.download("yolov8")
