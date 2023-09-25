#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# Define some environment variables
export IMAGE_NAME="generate-quiz"
export BASE_DIR=$(pwd)
export SECRETS_FILE_NAME=mega-ppp.json
export CONTAINER_SECRETS_DIR=secrets

# Build the image based on the Dockerfile
docker build -t $IMAGE_NAME -f Dockerfile .

# Run the container
docker run --rm --name $IMAGE_NAME \
-e GOOGLE_APPLICATION_CREDENTIALS=$CONTAINER_SECRETS_DIR/$SECRETS_FILE_NAME \
-ti $IMAGE_NAME