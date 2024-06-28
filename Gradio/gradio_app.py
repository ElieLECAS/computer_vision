import gradio as gr
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Charger le modèle YOLOv8 fine-tuné
model = YOLO('yolov8_fall_detection.pt')

# Ajouter un attribut pour les noms des classes
model.class_names = ['Fall-Detected']

# Fonction de prédiction
def predict(frame, conf_threshold=0.5):  # Seuil de confiance par défaut à 50%
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb)

    status = "Nothing detected"
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = box.conf[0]

            if confidence < conf_threshold:  # Appliquer le seuil de confiance
                continue

            if class_id < len(model.class_names):
                label = model.class_names[class_id]

                if label.lower() == 'fall-detected':  # Assurez-vous que le label pour la chute est correct
                    color = (0, 0, 255)  # Rouge pour les chutes
                    text = f"Fall detected: {confidence:.2f}"
                    status = "Fall detected"
                    # Dessiner le rectangle autour de l'objet détecté
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                else:
                    color = (0, 255, 0)  # Vert pour d'autres objets
                    text = f"{label}: {confidence:.2f}"
                    # Dessiner le rectangle autour de l'objet détecté
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return frame_pil, status

# Fonction pour capturer le flux vidéo en continu
def video_frame_generator(conf_threshold=0.6):
    cap = cv2.VideoCapture(0)  # 0 pour utiliser la webcam par défaut
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            yield None, "Webcam not detected"
            break
        frame_pil, status = predict(frame, conf_threshold)
        yield frame_pil, status

    cap.release()

# Créer l'interface Gradio
iface = gr.Interface(
    fn=video_frame_generator,
    inputs=None,
    outputs=[gr.Image(type="pil", label="Detection"), gr.Textbox(label="Status")],
    live=True,
    title="Detection de Chutes en Temps Réel",
    description="Utilisez votre webcam pour détecter les chutes en temps réel."
)

# Lancer l'interface
if __name__ == "__main__":
    iface.launch()
