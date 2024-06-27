import cv2
import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Charger le modèle YOLOv8 réentraîné
model = YOLO('../Model/yolov8_fall_detection.pt')

def detect_fall(frame):
    # Convertir l'image de BGR à RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Effectuer la prédiction
    results = model(frame_rgb)
    
    # Vérifier si une chute est détectée
    fall_detected = any([result['class'] == 0 for result in results[0].boxes.xywhn])
    
    # Dessiner les résultats sur l'image
    annotated_frame = results[0].plot()
    
    # Convertir l'image de RGB à BGR pour l'affichage OpenCV
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
    
    return annotated_frame, fall_detected

def process_video(video_frame):
    annotated_frame, fall_detected = detect_fall(video_frame)
    return (annotated_frame, "Fall Detected" if fall_detected else "No Fall Detected")

# Créer l'interface Gradio
webcam = gr.inputs.Image(shape=(640, 480), source="webcam", tool="editor", type="numpy")
output_image = gr.outputs.Image(type="numpy")
output_text = gr.outputs.Textbox()

gr.Interface(fn=process_video, inputs=webcam, outputs=[output_image, output_text],
             live=True, capture_session=True).launch()
