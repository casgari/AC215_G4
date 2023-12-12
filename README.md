# PAVVY: Learning Tools for Lecture Content

*Pavlos' Perceptron Pals (Cyrus Asgari, Ben Ray, Caleb Saul, Warren Sunada Wong, Chase Van Amburg)*

This repository was produced as part of the final project for Harvard’s AC215 Fall 2023 course. The purpose of this `README` is to explain to a developer how to utilize this repository, including usage of the individual containers that comprise the PAVVY app, the model training workflow, and deployment to a Kubernetes Cluster using Ansible. For a more accessible overview of the project, please see our Medium post [here](https://medium.com/@cbsaul/pavvy-learning-tools-for-lecture-content-aab450895ee1) and a video (including a live demo of PAVVY) [here](https://youtu.be/cfEjLLcUEgk?si=rVn4BaZa7n7fj4o8).

<img src="images/pavvy_logo.png" width="800">

## Introduction

PAVVY is an application that can process lecture videos or transcripts to produce keyphrases and generate quizzes, which allow learners to better review and engage with lecture content.

The deployed PAVVY application comprises of four main containers - video to audio preprocessing, audio transcription, keyword highlighting, quiz generation. These four containers can be run atomically from the `src/pipeline-workflow/` directory. In addition, we have a container that takes care of data versioning (found in the root of the repository) and a container that runs training for our keyword extraction model (found in `src/pipeline-workflow/model-training`).

We utilized several MLOps tools to optimize our data preprocessing and training workflows, including usage of TensorFlow Data, Dask for efficient transformations of lecture videos, support for serverless training with Vertex AI on multiple GPUs, and performance tracking with Weights & Biases. Upon completing training, our four main containers were formulated into a Vertex AI (Kubeflow) pipeline, allowing for effective orchestration of our app's components.

The trained model was deployed to an endpoint in Vertex AI, while the other containers were deployed to cloud functions and Cloud Run, accessible via an API. The final application comprises of a frontend built using React (found in `src/frontend-react`) and a backend API service using FastAPI to expose the models' functionality to the frontend (found in `src/api-service`). The entire application is then deployed to a Kubernetes Cluster using Ansible (using the files found in `src/deployment`).

The usage of these directories is explained in this `README` to allow a developer to replicate our steps. Despite the process being simple to follow, a developer is expected to have robust coding experience and some experience with MLOps to be able to fully interact with the external services (such as Google Cloud Platform) that are necessary for this project. Before digging into how to use this repository, it is helpful to spend some time familiarizing yourself with the following overview of the application design and project organization.

## Application Design
Full details of the application design, including the solution and technical architecture, can be found in `reports/application_design.pdf`. However, a useful schema for the four main containers in the project and the relationship between these containers and the Google Cloud Platform (GCP) Bucket can be seen here:

<img src="images/app_schema.png"  width="800">


## Project Organization
------------
    .
    ├── LICENSE
    ├── .dvc 
    ├── notebooks            <- Jupyter notebooks for EDA and model testing
    │   ├── intial_model_construction.ipynb
    │   └── tf_intial_model_construction_with_multigpu.ipynb
    ├── README.md
    ├── reports              <- Reports, midterm/final presentation, application design document
    │   ├── application_design.pdf
    │   ├── milestone2.pdf
    │   ├── milestone3.pdf
    │   ├── milestone4.md
    │   ├── milestone4_presentation.pdf
    │   ├── milestone5.pdf
    │   └── final_presentation.pdf
    ├── images               <- Folder containing images used in reports
    ├── cli.py               <- Files in the root of repo for data versioning
    ├── docker-shell.sh
    ├── Dockerfile
    ├── Pipfile
    ├── Pipfile.lock
    ├── requirements.txt
    ├── keyword_dataset.dvc
    └── src                  <- Source code and Dockerfiles for training and deployment
        ├── pipeline-workflow
        │   ├── model-deployment    <- Script to deploy and get predictions from the keyword extraction model
        │   │   └── ...
        │   ├── audio-transcription <- Use Whisper JAX for transcription
        │   │   └── ...
        │   ├── quiz-generation     <- Generate quizzes from transcribed text
        │   │   └── ...
        │   ├── data-conversion     <- Convert video to audio file
        │   │   └── ...
        │   └── model_training      <- Scripts for training keyword extraction model
        │       └── ...  
        ├── api-service     <- Code for backend
        │   ├── api                 <- Scripts that call cloud functions, cloud run, and Distilbert endpoint
        │   │   ├── model.py
        │   │   ├── service.py
        │   │   └── tracker.py
        │   ├── docker-shell.sh
        │   ├── docker-entrypoint.sh
        │   ├── Dockerfile
        │   ├── Pipfile
        │   └── Pipfile.lock
        ├── frontend-react <- Code for frontend
        │   └── src
        │   │   └── ...
        │   ├── docker-shell.sh
        │   ├── Dockerfile
        │   ├── Dockerfile.dev
        │   ├── package-lock.json
        │   ├── package.json
        │   └── yarn.lock
        └── deployment     <- Code for deployment to GCP with Ansible
            ├── cli.py
            ├── deploy-app.sh
            ├── deploy-create-instance.yml     <- Creates App Application Machine on GCP
            ├── deploy-docker-images.yml       <- Builds Docker images and pushes them to GCR
            ├── deploy-k8s-cluster.yml         <- Creates Kubernetes Cluster and deploys containers
            ├── deploy-provision-instance.yml  <- Configures app server instance
            ├── deploy-setup-containers.yml    <- Configures containers on app server
            ├── deploy-setup-webserver.yml     <- Configures webserver on the server instance
            ├── docker-entrypoint.sh
            ├── docker-shell.sh
            ├── Dockerfile
            ├── inventory.yml                  <- Defines global variables for Ansible
            └── run-ml-pipeline.sh

--------

## Individual Containers 
We have four different containers comprising the different components of our pipeline, which can be run locally or on the cloud. Here are the steps to get a container running: First, pick a container, for example `data-conversion`, then open up a terminal and run
```
cd src/pipeline-workflow/<CONTAINER NAME>
sh docker-shell.sh
```
The instructions for how to use each container are as follows:

### Video to Audio Preprocessing

`/data-conversion`

This container converts .mp4 files to .mp3 for later use using the MoviePy library. First it downloads input video files from the Cloud Bucket, then it converts them and uploads back to a different folder in the Cloud Bucket. The code uses Dask to run the preprocessing steps in parallel. From within the container, run:
```
python cli.py -c 
```

### Audio Transcription

`/audio-transcription`

This container transcribes lecture audio using OpenAI's Whisper JAX. To run, use
```
python transcribe.py
```
with flag `-d` to download audio files from the bucket, and `-t` to transcribe the downloaded files.

### Keyword Highlighting

`/model-deployment`

We have our trained DistilBERT model saved on GCP from our model training workflow (more on this below). To run, use
```
python cli.py
```
The `-p` flag runs a prediction using the VertexAI endpoint. To deploy a new version of the model, use the `-d` flag. Or to make a prediction locally, use `-l`.


### Quiz Generation

`/quiz-generation`

For quiz generation we make use of the OpenAI ChatGPT 3.5 API. This container takes in a lecture transcript in `.txt` format, then uses prompt engineering to get a quiz from ChatGPT. We generate the quiz with
```
python generate.py -g
```

## Model Training Workflow
The purpose of model training was to train an optimal derivation of the BERT model for usage in the Keyword Extraction container.

To optimize the training workflow, we take advantage of distributed computing by handling our data as TF Datasets, shuffling the data appropriately, and prefetching batches of the data for training, on multiple GPUs if available (look for the `model.prepare_tf_dataset` method in `src/pipeline-workflow/model-training/package/trainer/task.py`). Furthermore, we have fully implemented support for serverless training on multiple GPUs on Vertex AI, using TensorFlow's support for distributed training (look for `tf.distribute.MirroredStrategy()` in `src/pipeline-workflow/model-training/package/trainer/task.py`).

Having set up the workflow to handle data in a distributed way, we ran our serverless experiments with different versions and derivations of the BERT model (accessed via `TFAutoModelForTokenClassification`) to improve our performance on keyword extraction. The training data was a large set of labelled scientific abstracts obtained from [Inspec](https://huggingface.co/datasets/midas/inspec), where the labels (`B`, `I`, or `O`), represent a word being at the beginning, inside, or outside a keyword phrase. All experiments were run on a single A100 GPU on Vertex AI. We tracked our performance of our many training runs using Weights & Biases and stored our model artifacts on the platform.

We experimented with small models (e.g. `roberta-tiny-cased-trained`) which has only 28M parameters, and large models (e.g. the original BERT and RoBERTa models) which have over 100M parameters. Naturally, the smaller models performed inference faster but with a lower accuracy. We were able to find a useful compromise using model distillation. The distilled BERT and RoBERTa models achieved 99% of the accuracy achieved by the full-sized models with an acceptable inference time for deployment. Finally, we chose to deploy the `distilbert-base-uncased` model because of its very slight performance advantage over `distilroberta-base`. Below you can see the results of our serverless training experiments and an image of the successful serverless training job on Vertex AI.

![wnb image](images/wandb.png)
![training_results image](images/training_results.png)

![wnb image](images/serverless.jpeg)

To replicate this, the relevant files can be found in the `src/pipeline-workflow/model-training` directory with usage as follows:

(1) `src/pipeline-workflow/model_training/docker-shell.sh` - this script creates a container (defined in `src/pipeline-workflow/model_training/Dockerfile`) for running training in a standardized environment. Run with `sh docker-shell.sh`.

(2) `src/pipeline-workflow/model_training/package` - this package contains all the necessary scripts for running training, tracking performance via Weights & Biases, and saving the trained model artifact. Before running serverless training, this package is compiled by running `sh package-trainer.sh` inside the Docker container.

(3) `src/pipeline-workflow/model_training/cli.sh` - we use this script to submit the serverless job to Vertex AI, specifying parameters for training and the `ACCELERATOR_COUNT` for running training with multiple GPUs. Run with `sh cli.sh`.

## Application Design
### Backend API Service

We built a backend api service using fast API to expose model functionality to the frontend. Fast API gives us an interactive API documentation and exploration tool for free. An image of the documation tool is included below:

<img src="images/api_docs.png"  width="800">

`/predict` is called when users upload a lecture video to the frontend and wish to extract keywords and generate a quiz from it. `/predicttext` is used when users upload a lecture transcript to the frontend and wish to extract keywords and generate a quiz from it. These options are clear to see in the frontend below.

We can also easily test our APIs using this tool. Screenshots from successful tests of both `/predict` and `/predicttext` are included below:

<img src="images/predict_api_test.png"  width="800">

It is clear to see from this `/predict` testing that the server response is successful, with the response body returning keywords and a generated quiz as expected.

<img src="images/predicttext_api_test.png"  width="800">

A sucessful sever response is also observed from `/predicttext`.

The `api-service` container has all the files to run and expose the backend APIs.

To run the container locally:
- Open a terminal and navigate to `/src/api-service`
- Run `sh docker-shell.sh`
- Once inside the docker container run `uvicorn_server`
- To view and test APIs go to `http://localhost:9000/docs`

### Frontend Implementation

We have built a React app with Material UI framework to extract keywords and generate quizzes from lecture videos or lecture transcripts. Using the app, a user easily uploads their lecture video or text file. The app will send the video or text file through to the backend API to get the outputs. 

Here is a screenshot of our original version of the frontend:

<img src="images/homepage.png" width="800">

And here are a set of screenshots of the final frontend:
<img src="images/homepage_final.png" width="800">

The structure of the frontend is described by Material UI `<Container>` components, along with `<Typography>` and `<Button>` elements. Background images are custom .svg elements. File upload is the input that gets sent through the backend, and the Keyword and Quiz boxes are the output from the backend to the frontend. The `frontend-react` container contains all the files to develop and build our React app. 

Based on the Material UI framework, the `/app` folder contains specifications for high-level styling, such as for general `<Typography>` elements,  and overall structure of the application. The `index.js` file of the homepage holds the bulk of Pavvy's frontend code, under `/components/home`. Here, we connect with `api-service/service.py` to communicate with the backend.

To run the container locally:
- Open a terminal and navigate to `/src/frontend-react`
- Run `sh docker-shell.sh`
- If running the container for the first time, run `npm install`
- To include additional dependencies, run `npm install @emotion/react react-spinners`
- Once inside the docker container run `yarn start`
- Go to `http://localhost:3000` to access the app locally

Note that the above will only be hosted if the `api-service` container is running as well.

## Deployment to Kubernetes Cluster using Ansible.

**Ansible Usage For Automated Deployment**

We use Ansible to create, provision, and deploy our frontend and backend to GCP in an automated fashion. Ansible allows us to manage infrastructure as code, helping us keep track of our app infrastructure as code in GitHub. 

The `src/deployment` directory helps manage building and deploying all our app containers through Ansible, with all Docker images going to the Google Container Registry (GCR). To run the deployment process on your local device:

- Open a terminal and navigate to `/src/deployment`
- Run `sh docker-shell.sh`
- Build and push Docker Containers to GCR
```
ansible-playbook deploy-docker-images.yml -i inventory.yml
```

- Create Compute Instance (VM) in GCP
```
ansible-playbook deploy-create-instance.yml -i inventory.yml --extra-vars cluster_state=present
```

- Once the command runs successfully, get the IP address of the compute instance from GCP Console and update the `appserver>hosts` in `src/deployment/inventory.yml` file

- Provision Compute Instance in GCP, install and setup all the required things for deployment.
```
ansible-playbook deploy-provision-instance.yml -i inventory.yml
```

- Setup Docker Containers in the Compute Instance
```
ansible-playbook deploy-setup-containers.yml -i inventory.yml
```

- Setup Webserver on the Compute Instance
```
ansible-playbook deploy-setup-webserver.yml -i inventory.yml
```
Once the command runs go to `http://<External IP>/` to interact with the website, where `<External IP>` is shown in the GCP console and in the return statement from last step above. For reference, these images show our app deployed on a single VM in GCP and an image from the output of the last step of the process (taken on December 12, 2023):

<img src="images/vminstance.png"  width="800">
<img src="images/ansible_laststep.png"  width="800">

**Deployment to Kubernetes**

Kubernetes (K8s) is an open-source container orchestration system for automated scaling and management. We use Kubernetes to deploy our app on multiple servers with automatic load balancing and failovers. To create the Kubernetes Cluster, enable the relevant APIs and run the following code to start the container:

- Open a terminal and navigate to `/src/deployment`
- Run `sh docker-shell.sh`
- Build and push Docker Containers to GCR
```
ansible-playbook deploy-docker-images.yml -i inventory.yml
```
- Create and deploy the Cluster
```
ansible-playbook deploy-k8s-cluster.yml -i inventory.yml --extra-vars cluster_state=present
```
Once the command runs go to `http://<External IP>.sslip.io` to interact with the website, where `<External IP>` is shown in the GCP console and in the return statement from last step above. For reference, these images show our app deployed on a Kubernetes Cluster in GCP and an image showing part of the output from the last step of the process (taken on December 12, 2023):

<img src="images/kubernetes.png"  width="800">
<img src="images/kubernetesengine.png"  width="800">
<img src="images/kubernetes_laststep.png"  width="800">


