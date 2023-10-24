#!/bin/bash

set -e

export IMAGE_NAME=model-deployment-cli
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../secrets/
export GOOGLE_SECRETS_FILE=model-deployment.json
export WANDB_SECRETS_FILE=wandb_key.txt
export GCP_PROJECT="ac215-group-4"
export GCS_MODELS_BUCKET_NAME="keyword_models_mega_ppp" 
export CONTAINER_SECRETS_DIR=/secrets

# ACTION: May need to change above to mega-ppp/keyword_models


# Build the image based on the Dockerfile
#docker build -t $IMAGE_NAME -f Dockerfile .
# M1/2 chip macs use this line
docker buildx build -t $IMAGE_NAME --platform=linux/amd64 -f Dockerfile .

# Run Container
docker run --rm --name $IMAGE_NAME -ti \
-v "$BASE_DIR":/app \
-v "$SECRETS_DIR":$CONTAINER_SECRETS_DIR \
-e GOOGLE_APPLICATION_CREDENTIALS=/$CONTAINER_SECRETS_DIR/$GOOGLE_SECRETS_FILE \
-e WANDB_KEY=/$CONTAINER_SECRETS_DIR/$WANDB_SECRETS_FILE \
-e GCP_PROJECT=$GCP_PROJECT \
-e GCS_MODELS_BUCKET_NAME=$GCS_MODELS_BUCKET_NAME \
$IMAGE_NAME