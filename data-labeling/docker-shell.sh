#!/bin/bash

set -e

export GCS_BUCKET_NAME="mushroom-app-data-demo"
export GCP_PROJECT="ac215-project" [REPLACE WITH YOUR GCP PROJECT]
export GCP_ZONE="us-central1-a"

# Create the network if we don't have it yet
docker network inspect data-labeling-network >/dev/null 2>&1 || docker network create data-labeling-network

# Build the image based on the Dockerfile
docker build -t data-label-cli --platform=linux/arm64/v8 -f Dockerfile .

# Run All Containers
docker-compose run --rm --service-ports data-label-cli