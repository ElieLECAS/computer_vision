import gradio as gr
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

model = YOLO("yolov8_fall_detection.pt")

def video_feed(conf_threshold, iou_threshold):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not start camera.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(
            source=frame_rgb,
            conf=conf_threshold,
            iou=iou_threshold,
            show_labels=True,
            show_conf=True,
            imgsz=640,
        )
        
        logs = []
        logs.append(f"Confidence threshold: {conf_threshold}")
        logs.append(f"IoU threshold: {iou_threshold}")
        
        for r in results:
            detection_log = f"Detected objects ({len(r.boxes)}): "
            for box in r.boxes:
                detection_log += f"{box.cls} "
            detection_log += f"in {r.speed:.1f}ms"
            logs.append(detection_log)

            im_array = r.plot()
            im = Image.fromarray(im_array[..., ::-1])
            im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        
        ret, buffer = cv2.imencode('.jpg', im)
        frame = buffer.tobytes()
        
        yield (frame, "\n".join(logs))
    
    cap.release()

def run_video(conf_threshold, iou_threshold):
    return gr.Video(video_feed(conf_threshold, iou_threshold))

iface = gr.Interface(
    fn=run_video,
    inputs=[
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold"),
    ],
    outputs=[
        gr.Video(label="Result"),
        gr.Textbox(label="Logs", interactive=False)
    ],
    title="Ultralytics Real-time Detection",
    description="Real-time video feed with Ultralytics YOLOv8 model for object detection.",
    live=True
)

if __name__ == "__main__":
    iface.launch()
