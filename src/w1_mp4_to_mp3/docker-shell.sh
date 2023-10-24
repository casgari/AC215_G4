#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# Define some environment variables
export IMAGE_NAME="preprocess_audio_file"
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../secrets
export SECRETS_FILE_NAME=ml-workflow.json
export GCP_PROJECT="ac215-group-4"
export GCS_BUCKET_NAME="mega-ppp-ml-workflow"
export CONTAINER_SECRETS_DIR=/secrets

# Build the image based on the Dockerfile
docker build -t $IMAGE_NAME -f Dockerfile .

# Run the container
docker run --rm --name $IMAGE_NAME -ti \
--mount type=bind,source="$BASE_DIR",target=/app \
--mount type=bind,source="$SECRETS_DIR",target=$CONTAINER_SECRETS_DIR \
-e GOOGLE_APPLICATION_CREDENTIALS=$CONTAINER_SECRETS_DIR/$SECRETS_FILE_NAME \
-e GCP_PROJECT=$GCP_PROJECT \
-e GCS_BUCKET_NAME=$GCS_BUCKET_NAME \
$IMAGE_NAME