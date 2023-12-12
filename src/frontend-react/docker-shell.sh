#!/bin/bash

set -e

export IMAGE_NAME="ppp-app-frontend-react"

# Build the image based on the Dockerfile
# docker build -t $IMAGE_NAME -f Dockerfile.dev .
# M1/2 chip macs use this line
docker build -t $IMAGE_NAME -f Dockerfile.dev .

docker run --rm --name $IMAGE_NAME -ti -v "$(pwd)/:/app/" -p 3000:3000 $IMAGE_NAME