#!/bin/bash

# Configuración de variables
IMAGE_NAME=${IMAGE_NAME:-"condition-predictor"}
IMAGE_TAG=${DOCKER_IMAGE_TAG:-'dev'}
VOLUME=$(pwd):/code  # Mapea el directorio actual al contenedor

# Ejecutar Black para verificar el formato del código en el directorio actual
echo "Running Black linter..."
docker run -v "$VOLUME" --rm "$IMAGE_NAME":"$IMAGE_TAG" black --check --diff --color .

# Ejecutar Flake8 para verificar errores de estilo de código en el directorio actual
echo "Running Flake8 linter..."
docker run -v "$VOLUME" --rm "$IMAGE_NAME":"$IMAGE_TAG" flake8 .
