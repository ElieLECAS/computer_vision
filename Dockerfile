# Utiliser une image de base officielle avec Python et pip
FROM python:3.9-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers requirements.txt et le modèle YOLO dans le conteneur
COPY requirements.txt .
COPY yolov8_fall_detection2.pt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le répertoire Gradio dans le conteneur
COPY Gradio/ ./Gradio/

# Exposer le port 7860 pour Gradio
EXPOSE 7860

# Commande pour lancer l'application Gradio
CMD ["python", "Gradio/gradio_app.py"]
