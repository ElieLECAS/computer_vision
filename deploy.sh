#!/bin/bash

# Charger les variables d'environnement depuis le fichier .env
export $(grep -v '^#' .env | xargs)

# Nom de l'image Docker
IMAGE_NAME=gradio_app

# Construire l'image Docker
docker build -t $DOCKERHUB_USERNAME/$IMAGE_NAME .

# Se connecter à Docker Hub
echo $DOCKERHUB_PASSWORD | docker login -u $DOCKERHUB_USERNAME --password-stdin

# Taguer l'image Docker
docker tag $DOCKERHUB_USERNAME/$IMAGE_NAME $DOCKERHUB_USERNAME/$IMAGE_NAME:latest

# Pousser l'image Docker vers Docker Hub
docker push $DOCKERHUB_USERNAME/$IMAGE_NAME:latest

# Se connecter à Azure
az login

# Définir les variables Azure
RESOURCE_GROUP=RG_ABADIF
APP_SERVICE_PLAN=myAppServicePlan
WEBAPP_NAME=myGradioApp

# Créer un plan App Service (si ce n'est pas déjà fait)
az appservice plan create --name $APP_SERVICE_PLAN --resource-group $RESOURCE_GROUP --sku B1 --is-linux

# Créer une application web et déployer le conteneur
az webapp create --resource-group $RESOURCE_GROUP --plan $APP_SERVICE_PLAN --name $WEBAPP_NAME --deployment-container-image-name $DOCKERHUB_USERNAME/$IMAGE_NAME:latest

# Configurer le plan App Service pour tirer les images de Docker Hub
az webapp config container set --name $WEBAPP_NAME --resource-group $RESOURCE_GROUP --docker-custom-image-name $DOCKERHUB_USERNAME/$IMAGE_NAME:latest
