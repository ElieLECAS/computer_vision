import cv2
import gradio as gr
from ultralytics import YOLO

# Charger le modèle réentraîné pour la détection des chutes
fall_detection_model = YOLO('Gradio/best.torchscript')

# Imprimer les classes avec leurs index
class_names = fall_detection_model.names
for index, class_name in class_names.items():
    print(f"Index: {index}, Class: {class_name}")

def detect_fall(frame):
    fall_conf_threshold = 0.7  # Seuil de confiance pour la détection des chutes

    # Convertir l'image de BGR à RGB pour la détection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Appliquer le modèle de détection de chutes sur l'image entière
    results_fall = fall_detection_model(frame_rgb)

    fall_detected = False

    for result in results_fall:
        for fall_box in result.boxes:
            if fall_box.conf.item() >= fall_conf_threshold:  # Seulement si la confiance est élevée
                fall_detected = True
                x1, y1, x2, y2 = map(int, fall_box.xyxy[0])
                frame_rgb = cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)
                frame_rgb = cv2.putText(frame_rgb, f'Fall Detected: {fall_box.conf.item():.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Ajouter un texte pour indiquer si une chute est détectée
    status_text = "Fall Detected!" if fall_detected else "No Fall Detected"

    # Convertir l'image de RGB à BGR pour l'affichage dans Gradio
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # Retourner le cadre annoté et le texte de statut
    return frame_bgr, status_text

# Fonction pour capturer le flux vidéo et appliquer la détection
def video_stream():
    cap = cv2.VideoCapture(0)  # 0 pour la webcam intégrée, changez le numéro pour une autre caméra
    while cap.isOpened():
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
