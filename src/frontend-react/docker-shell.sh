#!/bin/bash

set -e

export IMAGE_NAME="ppp-app-frontend-react"

docker build -t $IMAGE_NAME --platform=linux/amd64/v2 -f Dockerfile.dev .
docker run --rm --name $IMAGE_NAME -ti -v "$(pwd)/:/app/" -p 3000:3000 $IMAGE_NAME