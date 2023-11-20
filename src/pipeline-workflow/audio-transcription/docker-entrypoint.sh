#!/bin/bash

echo "Container is running!!!"

args="$@"
echo $args

echo $CLOUDRUN

if [ "${CLOUDRUN}" = 1 ];
then
  pipenv run functions-framework --target transcribe_http
else
  if [[ -z ${args} ]]; 
  then
      # Authenticate gcloud using service account
      gcloud auth activate-service-account --key-file $GOOGLE_APPLICATION_CREDENTIALS
      # Set GCP Project Details
      gcloud config set project $GCP_PROJECT
      #/bin/bash
      pipenv shell
  else
    pipenv run python $args
  fi
fi