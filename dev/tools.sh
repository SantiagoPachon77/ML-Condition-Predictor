#!/bin/bash

# Configura el nombre y la etiqueta de la imagen de Docker
IMAGE_NAME=${IMAGE_NAME:-"condition-predictor"}
IMAGE_TAG=${DOCKER_IMAGE_TAG:-'dev'}

# Recibe el primer argumento como comando
cmd=$1
shift

# Define las acciones basadas en el comando proporcionado
case ${cmd} in

    b | build)
        echo "Build Docker image: ${IMAGE_NAME}..."
        # Construir la imagen de Docker sin argumentos de Nexus
        docker build -t "${IMAGE_NAME}":"${IMAGE_TAG}" .
    ;;

    l | linters)
        echo "Run linters: ${IMAGE_NAME}"
        # Ejecuta el script de linters
        dev/linters.sh  # Asegúrate de que el archivo esté en el directorio correcto
    ;;

    *)
        echo "Bad command. Options are:"
        grep -E "^    . \| .*\)$" "$0"
    ;;

esac
