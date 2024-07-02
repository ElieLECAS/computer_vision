import cv2
import gradio as gr
from ultralytics import YOLO
import time
import os
import pygame

# Charger le modèle YOLOv8 réentraîné pour la détection des chutes
fall_detection_model = YOLO('Gradio/best.torchscript')

# Initialiser les variables pour le suivi des chutes
fall_detected_time = None
fall_alert_triggered = False

# Initialiser pygame pour le son
pygame.mixer.init()
alert_sound_path = os.path.abspath('Gradio/alerte.mp3')

def play_alert_sound():
    pygame.mixer.music.load(alert_sound_path)
    pygame.mixer.music.play(-1)  # Répéter le son indéfiniment

def stop_alert_sound():
    pygame.mixer.music.stop()

def detect_fall(frame):
    global fall_detected_time, fall_alert_triggered
    conf_threshold = 0.70  # Seuil de confiance pour la détection des chutes

    # Convertir l'image de BGR à RGB pour le modèle YOLO
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convertir l'image en format compatible avec YOLO
    results = fall_detection_model(frame_rgb)

    fall_detected = False

    # Filtrer uniquement les détections avec une confiance suffisante
    for result in results:
        for box in result.boxes:
            if box.conf.item() >= conf_threshold:  # Condition de confiance corrigée
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf.item()  # Convertir le tensor en une valeur numérique
                label = f'Fall Detected: {conf:.2f}'
                frame_rgb = cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)
                frame_rgb = cv2.putText(frame_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                fall_detected = True

                # Détecter une chute et gérer le suivi du temps
                if fall_detected_time is None:
                    fall_detected_time = time.time()
                else:
                    elapsed_time = time.time() - fall_detected_time
                    if elapsed_time >= 5 and not fall_alert_triggered:
                        # Déclencher une alerte ou une sonnerie
                        print("Alerte : chute détectée pendant 5 secondes !")
                        play_alert_sound()
                        fall_alert_triggered = True

                break

    if not fall_detected:
        fall_detected_time = None
        if fall_alert_triggered:
            stop_alert_sound()
        fall_alert_triggered = False

    # Ajouter un texte pour indiquer si une chute est détectée
    status_text = "Fall Detected!" if fall_detected else "No Fall Detected"

    # Retourner le cadre annoté en RGB et le texte de statut
    return frame_rgb, status_text

# Fonction pour capturer le flux vidéo et appliquer la détection
def video_stream():
    cap = cv2.VideoCapture(2)  # 0 pour la webcam intégrée, changez le numéro pour une autre caméra
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
