import cv2
import gradio as gr
from ultralytics import YOLO

# Charger le modèle YOLOv8 pré-entraîné pour la détection des personnes
person_model = YOLO('yolov8n.pt')

# Charger le modèle réentraîné pour la détection des chutes
fall_detection_model = YOLO('yolov8_fall_detection.pt')

def detect_fall(frame):
    person_conf_threshold = 0.5  # Seuil de confiance pour la détection des personnes
    fall_conf_threshold = 0.5  # Seuil de confiance pour la détection des chutes

    # Convertir l'image en format compatible avec YOLO
    results_person = person_model(frame)
    
    fall_detected = False

    # Filtrer uniquement les détections de personnes avec une confiance suffisante
    for result in results_person:
        for box in result.boxes:
            if box.cls == 0 and box.conf.item() >= person_conf_threshold:  # 0 est l'index de la classe "personne" dans le modèle YOLOv8
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf.item()  # Convertir le tensor en une valeur numérique
                label = f'Person: {conf:.2f}'
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                frame = cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                # Extraire la région de la personne détectée
                person_roi = frame[y1:y2, x1:x2]

                # Appliquer le modèle de détection de chutes sur la région de la personne
                results_fall = fall_detection_model(person_roi)
                for fall_box in results_fall:
                    for fall_box in fall_box.boxes:
                        if fall_box.conf.item() >= fall_conf_threshold:  # Seulement si la confiance est élevée
                            fall_detected = True
                            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            frame = cv2.putText(frame, f'Fall Detected: {fall_box.conf.item():.2f}', (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Ajouter un texte pour indiquer si une chute est détectée
    status_text = "Fall Detected!" if fall_detected else "No Fall Detected"
    
    # Retourner le cadre annoté et le texte de statut
    return frame, status_text

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
