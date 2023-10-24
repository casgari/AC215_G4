#!/bin/bash

set -e

export IMAGE_NAME=mushroom-app-training-cli
export BASE_DIR=$(pwd)
export PERSISTENT_DIR=$(pwd)/../../../persistent-folder/
export SECRETS_DIR=$(pwd)/../../../secrets/
export GCP_PROJECT="ac215-project"
export GCS_BUCKET_NAME="vertex-tragedy"
export GCP_REGION="us-central1"
export GCS_PACKAGE_URI="gs://vertex-tragedy"


# Build the image based on the Dockerfile
#docker build -t $IMAGE_NAME -f Dockerfile .
# M1/2 chip macs use this line
docker build -t $IMAGE_NAME --platform=linux/arm64/v8 -f Dockerfile .

# Run Container
docker run --rm --name $IMAGE_NAME -ti \
-v "$BASE_DIR":/app \
-v "$SECRETS_DIR":/secrets \
-v "$PERSISTENT_DIR":/persistent \
-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/model-trainer.json \
-e GCP_PROJECT=$GCP_PROJECT \
-e GCS_BUCKET_NAME=$GCS_BUCKET_NAME \
-e GCP_REGION=$GCP_REGION \
-e GCS_PACKAGE_URI=$GCS_PACKAGE_URI \
$IMAGE_NAME