# Mushroom App: Model Deployment Demo

In this tutorial we will deploy a model to Vertex AI:
<img src="images/serverless-model-deployment.png"  width="800">

## Prerequisites
* Have Docker installed
* Cloned this repository to your local machine with a terminal up and running
* Check that your Docker is running with the following command

`docker run hello-world`

### Install Docker 
Install `Docker Desktop`

#### Ensure Docker Memory
- To make sure we can run multiple container go to Docker>Preferences>Resources and in "Memory" make sure you have selected > 4GB

### Install VSCode  
Follow the [instructions](https://code.visualstudio.com/download) for your operating system.  
If you already have a preferred text editor, skip this step.  

## Setup Environments
In this tutorial we will setup a container to manage building and deploying models to Vertex AI Model Registry and Model Endpoints.

**In order to complete this tutorial you will need your GCP account setup and a WandB account setup.**

### Clone the github repository
- Clone or download from [here](https://github.com/dlops-io/model-deployment)

### API's to enable in GCP for Project
Search for each of these in the GCP search bar and click enable to enable these API's
* Vertex AI API

### Setup GCP Credentials
Next step is to enable our container to have access to Storage buckets & Vertex AI(AI Platform) in  GCP. 

#### Create a local **secrets** folder

It is important to note that we do not want any secure information in Git. So we will manage these files outside of the git folder. At the same level as the `model-deployment` folder create a folder called **secrets**

Your folder structure should look like this:
```
   |-model-deployment
   |-secrets
```

#### Setup GCP Service Account
- Here are the step to create a service account:
- To setup a service account you will need to go to [GCP Console](https://console.cloud.google.com/home/dashboard), search for  "Service accounts" from the top search box. or go to: "IAM & Admins" > "Service accounts" from the top-left menu and create a new service account called "model-deployment". For "Service account permissions" select "Storage Admin", "AI Platform Admin", "Vertex AI Administrator".
- This will create a service account
- On the right "Actions" column click the vertical ... and select "Manage keys". A prompt for Create private key for "model-deployment" will appear select "JSON" and click create. This will download a Private key json file to your computer. Copy this json file into the **secrets** folder. Rename the json file to `model-deployment.json`

### Create GCS Bucket

We need a bucket to store the saved model files that we will be used by Vertext AI to deploy models.

- Go to `https://console.cloud.google.com/storage/browser`
- Create a bucket `mushroom-app-models-demo` [REPLACE WITH YOUR BUCKET NAME]

## Run Container

### Run `docker-shell.sh` or `docker-shell.bat`
Based on your OS, run the startup script to make building & running the container easy

This is what your `docker-shell` file will look like:
```
export IMAGE_NAME=model-deployment-cli
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../secrets/
export GCP_PROJECT="ac215-project" [REPLACE WITH YOUR PROJECT]
export GCS_MODELS_BUCKET_NAME="mushroom-app-models-demo" [REPLACE WITH YOUR BUCKET NAME]


# Build the image based on the Dockerfile
#docker build -t $IMAGE_NAME -f Dockerfile .
# M1/2 chip macs use this line
docker build -t $IMAGE_NAME --platform=linux/arm64/v8 -f Dockerfile .

# Run Container
docker run --rm --name $IMAGE_NAME -ti \
-v "$BASE_DIR":/app \
-v "$SECRETS_DIR":/secrets \
-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/model-deployment.json \
-e GCP_PROJECT=$GCP_PROJECT \
-e GCS_MODELS_BUCKET_NAME=$GCS_MODELS_BUCKET_NAME \
$IMAGE_NAME
```

- Make sure you are inside the `model-deployment` folder and open a terminal at this location
- Run `sh docker-shell.sh` or `docker-shell.bat` for windows

### Prepare Model for Deployment
We have our model weights stored in WandB after we performed serverless training. In this step we will download the model and upload it to a GCS bucket so Vertex AI can have access to it to deploy to an endpoint.

* Run `python cli.py --upload`, this will download the model weights from WandB and upload to the specified bucket in `GCS_MODELS_BUCKET_NAME`

### Upload & Deploy Model to Vertex AI
In this step we first upload our model to Vertex AI Model registry. Then we deploy the model as an endpoint in Vertex AI Online prediction.

* Run `python cli.py --deploy`, this option will both upload and deploy model to Vertex AI
* This will take a few minutes to complete
* Once the model has been deployed the endpoint will be displayed. The endpoint will be similar to: `projects/129349313346/locations/us-central1/endpoints/5072058134046965760`

### Test Predictions

* Update the endpoint uri in `cli.py`
* Run `python cli.py --predict`
* You  shouls see results simsialr to this:
```
Predict using endpoint
image_files: ['data/oyster_3.jpg', 'data/oyster_2.jpg', 'data/oyster_1.jpg', 'data/oyster_4.jpg', 'data/crimini_1.jpg']
Image: data/amanita_2.jpg
Result: Prediction(predictions=[[0.0887121782, 0.0439011417, 0.867386699]], deployed_model_id='3704450387047088128', model_version_id='1', model_resource_name='projects/129349313346/locations/us-central1/models/8243511463436615680', explanations=None)
[0.0887121782, 0.0439011417, 0.867386699] 2
Label:    amanita 

Image: data/oyster_4.jpg
Result: Prediction(predictions=[[0.986440122, 0.00689249625, 0.0066674049]], deployed_model_id='3704450387047088128', model_version_id='1', model_resource_name='projects/129349313346/locations/us-central1/models/8243511463436615680', explanations=None)
[0.986440122, 0.00689249625, 0.0066674049] 0
Label:    oyster 

Image: data/oyster_2.jpg
Result: Prediction(predictions=[[0.80594486, 0.0182529744, 0.175802067]], deployed_model_id='3704450387047088128', model_version_id='1', model_resource_name='projects/129349313346/locations/us-central1/models/8243511463436615680', explanations=None)
[0.80594486, 0.0182529744, 0.175802067] 0
Label:    oyster 
```