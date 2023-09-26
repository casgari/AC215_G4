
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../secrets/

REM Create the network if we don't have it yet
docker network inspect data-versioning-network >/dev/null 2>&1 || docker network create data-versioning-network

REM Build the image based on the Dockerfile
docker build -t data-version-cli -f Dockerfile .

REM Run Container
docker run --rm --name data-version-cli -ti \
-v "$BASE_DIR":/app \
-v "$SECRETS_DIR":/secrets \
-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/data-service-account.json \
-e GCP_PROJECT="ac215-project" \
-e GCP_ZONE="us-central1-a" \
--network data-versioning-network data-version-cli