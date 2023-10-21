#!/bin/bash

# set -e

export IMAGE_NAME="mushroom-app-workflow"
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../../../secrets/
export GCP_PROJECT="ac215-project"
export GCS_BUCKET_NAME="mushroom-app-ml-workflow-demo"
export GCS_SERVICE_ACCOUNT="ml-workflow@ac215-project.iam.gserviceaccount.com"

# Build the image based on the Dockerfile
#docker build -t $IMAGE_NAME -f Dockerfile .
docker build -t $IMAGE_NAME --platform=linux/amd64 -f Dockerfile .


# Run Container
docker run --rm --name $IMAGE_NAME -ti \
-v /var/run/docker.sock:/var/run/docker.sock \
-v "$BASE_DIR":/app \
-v "$SECRETS_DIR":/secrets \
-v "$BASE_DIR/../data-collector":/data-collector \
-v "$BASE_DIR/../data-processor":/data-processor \
-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/ml-workflow.json \
-e GCP_PROJECT=$GCP_PROJECT \
-e GCS_BUCKET_NAME=$GCS_BUCKET_NAME \
-e GCS_SERVICE_ACCOUNT=$GCS_SERVICE_ACCOUNT \
$IMAGE_NAME

