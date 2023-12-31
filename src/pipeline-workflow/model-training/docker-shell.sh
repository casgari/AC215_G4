#!/bin/bash

set -e

export IMAGE_NAME=model-training-cli
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../../../secrets/
export GCP_PROJECT="ac215-group-4"
export GCS_BUCKET_NAME="vertex-tragedy"
export GCP_REGION="us-central1"
export GCS_PACKAGE_URI="gs://vertex-tragedy"
export WANDB_KEY="2c2b43011dfdd34ca7eab2f22d19f997b4eec748"

# Build the image based on the Dockerfile
docker build -t $IMAGE_NAME --platform=linux/arm64/v8 -f Dockerfile .

# Run Container
docker run --rm --name $IMAGE_NAME -ti \
-v "$BASE_DIR":/app \
-v "$SECRETS_DIR":/secrets \
-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/ml-workflow.json \
-e GCP_PROJECT=$GCP_PROJECT \
-e GCS_BUCKET_NAME=$GCS_BUCKET_NAME \
-e GCP_REGION=$GCP_REGION \
-e GCS_PACKAGE_URI=$GCS_PACKAGE_URI \
-e WANDB_KEY=$WANDB_KEY \
$IMAGE_NAME