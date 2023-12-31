#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# Define some environment variables
export IMAGE_NAME="mega-ppp-quiz-generation"
export BASE_DIR=$(pwd)
export PERSISTENT_DIR=$(pwd)/../../../persistent-folder/
export SECRETS_DIR=$(pwd)/../../../secrets/
export GCP_PROJECT="ac215-group-4"
export GCS_BUCKET_NAME="mega-ppp-ml-workflow"


# Build the image based on the Dockerfile
docker build -t $IMAGE_NAME -f Dockerfile .

# Run the container
docker run --rm --name $IMAGE_NAME -ti \
-v "$BASE_DIR":/app \
-v "$SECRETS_DIR":/secrets \
-v "$PERSISTENT_DIR":/persistent \
-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/mega-ppp.json \
-e OPENAI_KEY=/secrets/openai_api_key.txt \
-e GCP_PROJECT=$GCP_PROJECT \
-e GCS_BUCKET_NAME=$GCS_BUCKET_NAME \
$IMAGE_NAME


