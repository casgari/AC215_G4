#!/bin/bash

set -e

export IMAGE_NAME="mega-ppp-workflow"
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../secrets
export SECRETS_FILE_NAME=ml-workflow.json
export GCP_PROJECT="ac215-group-4"
export GCS_BUCKET_NAME="mega-ppp-ml-workflow"
export GCS_SERVICE_ACCOUNT="ml-workflow-135@ac215-group-4.iam.gserviceaccount.com"
export CONTAINER_SECRETS_DIR=/secrets

# Build the image based on the Dockerfile
docker build -t $IMAGE_NAME --platform=linux/amd64 -f Dockerfile .


# Run Container
docker run --rm --name $IMAGE_NAME -ti \
-v /var/run/docker.sock:/var/run/docker.sock \
-v "$BASE_DIR":/app \
--mount type=bind,source="$SECRETS_DIR",target=$CONTAINER_SECRETS_DIR \
-v "$BASE_DIR/../data-collector":/data-collector \
-v "$BASE_DIR/../data-processor":/data-processor \
-e GOOGLE_APPLICATION_CREDENTIALS=$CONTAINER_SECRETS_DIR/$SECRETS_FILE_NAME \
-e GCP_PROJECT=$GCP_PROJECT \
-e GCS_BUCKET_NAME=$GCS_BUCKET_NAME \
-e GCS_SERVICE_ACCOUNT=$GCS_SERVICE_ACCOUNT \
$IMAGE_NAME
