import cv2
import gradio as gr
from ultralytics import YOLO

# Charger le modèle YOLOv8 réentraîné pour la détection des chutes
fall_detection_model = YOLO('Gradio/best.torchscript')

def detect_fall(frame):
    conf_threshold = 0.65  # Seuil de confiance pour la détection des chutes

    # Convertir l'image de BGR à RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convertir l'image en format compatible avec YOLO
    results = fall_detection_model(frame_rgb)

    fall_detected = False

    # Filtrer uniquement les détections avec une confiance suffisante
    for result in results:
        for box in result.boxes:
            if 0.7 < box.conf.item() >= conf_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf.item()  # Convertir le tensor en une valeur numérique
                label = f'Fall Detected: {conf:.2f}'
                frame_rgb = cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)
                frame_rgb = cv2.putText(frame_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                fall_detected = True

    # Ajouter un texte pour indiquer si une chute est détectée
    status_text = "Fall Detected!" if fall_detected else "No Fall Detected"

# Retourner le cadre annoté et le texte de statut
    return frame_rgb, status_text

# Fonction pour capturer le flux vidéo et appliquer la détection
def video_stream():
    cap = cv2.VideoCapture(0)  # 0 pour la webcam intégrée, changez le numéro pour une autre caméra
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, status_text = detect_fall(frame)
        yield frame, status_text

    cap.release()

# Créer l'interface Gradio
gr.Interface(fn=video_stream, 
             inputs=[], 
             outputs=[gr.Image(type="numpy", label="Webcam Feed"), gr.Textbox(label="Status")], 
             live=True).launch()