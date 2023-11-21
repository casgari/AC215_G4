AC215 - Milestone5

Project Organization
------------
    .
    ├── LICENSE
    ├── .dvc 
    ├── notebooks            <- Jupyter notebooks for EDA and model testing
    │   ├── intial_model_construction.ipynb
    │   └── tf_intial_model_construction_with_multigpu.ipynb
    ├── README.md
    ├── reports              <- Folder containing past milestone markdown submissions
    │   └── milestone2.md
    │   └── milestone3.md
    ├── images               <- Folder containing wandb image
    │   └── wandb.png
    ├── cli.py               <- Files in the root of repo for data versioning
    ├── docker-shell.sh
    ├── Dockerfile
    ├── Pipfile
    ├── Pipfile.lock
    ├── requirements.txt
    ├── keyword_dataset.dvc
    └── src                  <- Source code and Dockerfiles for data processing and modeling
        ├── individual-containers <- initial iteration of containers to be run independently of the pipeline, from earlier milestones.
        └── pipeline-workflow
            ├── model-deployment <- Script to deploy and get predictions from the keyword extraction model
            │   ├── Dockerfile
            │   ├── Pipfile
            │   ├── Pipfile.lock
            │   ├── docker-shell.sh
            │   ├── docker-entrypoint.sh
            │   └── cli.py
            ├── audio-transcription <- Use Whisper JAX for transcription
            │   ├── Dockerfile
            │   ├── Pipfile
            │   ├── Pipfile.lock
            │   ├── docker-shell.sh
            │   ├── docker-entrypoint.sh
            │   └── cli.py
            ├── quiz-generation  <- generate quizzes from transcribed text
            │   ├── Dockerfile
            │   ├── Pipfile
            │   ├── Pipfile.lock
            │   ├── docker-shell.sh
            │   ├── docker-entrypoint.sh
            │   └── cli.py
            ├── data-conversion <- Convert video to audio file
            │   ├── Dockerfile
            │   ├── Pipfile
            │   ├── Pipfile.lock
            │   ├── docker-shell.sh
            │   ├── docker-entrypoint.sh
            │   └── cli.py
            └── model_training    <- Scripts for training keyword extraction model
                └── package
                └── Dockerfile
                ├── Pipfile
                ├── Pipfile.lock
                ├── docker-shell.sh
                ├── docker-entrypoint.sh
                ├── package-trainer.sh
                ├── cli.sh
                ├── cli.py
                └── package
                    ├── PKG-INFO
                    ├── setup.cfg
                    ├── setup.py
                    └── trainer
                        ├── task.py
                        └── __init__.py

            
                

--------
# AC215 - Milestone5 - Learning Tools for Transcribed Lecture Audio

**Team Members**
Cyrus Asgari, Ben Ray, Caleb Saul, Warren Sunada Wong, Chase Van Amburg

**Group Name**
Pavlos' Perceptron Pals

**Project**
In this project we aim to develop an application that can process lecture videos to generate transcripts with key-word highlighting and offer auto-generated quizzes with both questions and answers.

**Recap of work to date (see `reports/` for more details)**
We built the four main containers - video to audio preprocessing, audio transcription, keyword highlighting, quiz generation - that are deployed in our pipeline. These can be run atomically from the `individual-containers/` directory. In addition, we have a container that takes care of data versioning (found in the root of the repository) and a container that runs training for our keyword extraction model (found in `pipeline-workflow/model-training`). 

We have utilized several advanced tools to optimize our data preprocessing and training workflows, including usage of TensorFlow Data, Dask for efficient transformations of lecture videos, support for serverless training with Vertex AI on multiple GPUs, and performance tracking with Weights & Biases. Our training efforts were primarily focused on training our keyword extraction model, using different derivations of the BERT model. All experiments were run on a single A100 GPU on Vertex AI, although we have also fully implemented and tested support for training on multiple GPUs, using TensorFlow's support for distributed training (i.e. `tf.distribute.MirroredStrategy()`). We chose to deploy the `distilbert-base-uncased` model, which performed well in terms of inference time and achieved 99% of the performance of the undistilled `bert-base-uncased` model.

Upon completing training, our four main containers were formulated into a Vertex AI (Kubeflow) pipeline, allowing for effective orchestration of our app's components and usage of cloud functions in Milestone 5.

### Milestone 5 ###

In this milestone, we have focused on the development and deployment of a backend API service and the client-side of the application. Before implementing this, we created a detailed design document outlining the application’s Solution and Technical architectures, which can be found in `reports/application_design.pdf`.

Our backend leverages many of the tools made accessible by AC215. Our [video to audio preprocessing container](https://us-central1-ac215-group-4.cloudfunctions.net/data-preprocessing) and [quiz generation container](https://us-central1-ac215-group-4.cloudfunctions.net/quiz-generation) are deployed as cloud functions on GCP, since they are relatively lightweight. By comparison, our audio transcription container, which calls the Whisper-Jax model, takes far longer to run, and so benefits further optimization within its own Docker container. Thus, we chose to deploy the [audio transcription container](https://audio-transcription-hcsan6rz2q-uc.a.run.app/) in Cloud Run. Finally, as completed in class, we deployed our trained DistilBERT model from the keyword extraction container to its own endpoint on GCP. This allows for efficient updates of this model in future, without needing to redploy an entire container image on Cloud Run or a new cloud function.

Our frontend interface is built in React and is contained within the `src/frontend-react` directory. At this stage, the frontend represents a minimum viable product for a user. This means that a user can upload a video and receive both the keywords and the generated quiz, but they have limited ability to re-format the outputs (e.g. as a highlighted transcript, instead of a list of keywords). These features that make the frontend more user-friendly will follow in Milestone 6.

Finally, we 


In the remainder of this update, we will explain the code structure of the [deliverables](https://harvard-iacs.github.io/2023-AC215/milestone5/) for Milestone 5. Given the size of the codebase, please note that we reserve this space for describing the deliverables for Milestone 5 only, and usage of other parts of the codebase is desribed in earlier reports (see `reports/`).

#### Code Structure

**Machine Learning Workflow Implementation**

Having completed preprocessing, we train our model using the `src/pipeline-workflow/model_training` container with usage as follows:

(1) `src/model_training/docker-shell.sh` - this script creates a container (defined in `src/pipeline-workflow/model_training/Dockerfile`) for running training in a standardized environment. Run with `sh docker-shell.sh`.

(2) `src/pipeline-workflow/model_training/package` - this package contains all the necessary scripts for running training, tracking performance via Weights & Biases, and saving the trained model artifact. Before running serverless training, this package is compiled by running `sh package-trainer.sh` inside the Docker container.

(3) `src/pipeline-workflow/model_training/cli.sh` - we use this script to submit the serverless job to Vertex AI, specifying parameters for training and the `ACCELERATOR_COUNT` for running training with multiple GPUs. Run with `sh cli.sh`.

Below you can see the results of our serverless training experiments on an A100 chip. We experimented with small models (e.g. `roberta-tiny-cased-trained`) which has only 28M parameters, and large models (e.g. the original BERT and RoBERTa models) which have over 100M parameters. Naturally, the smaller models performed inference faster but with a lower accuracy. We were able to find a useful compromise using model distillation. The distilled BERT and RoBERTa models achieved 99% of the accuracy achieved by the full-sized models with an acceptable inference time for deployment. Finally, we chose to deploy the `distilbert-base-uncased` model because of its very slight performance advantage over `distilroberta-base`.

![wnb image](images/wandb.png)
![training_results image](images/training_results.png)

Below we include an image of the successful serverless training job on Vertex AI.

![wnb image](images/serverless.jpeg)


**Vertex AI (Kubeflow) Pipeline Implementation**

Below you can see two images related to our work with Vertex AI Pipelines. The first showcases our models which we've saved within the Vertex AI Model Registry, and the second is our Endpoints, which are our models that are available for model prediction requests.

![registry image](images/registry.png)
![endpoint image](images/endpoint.png)


We designed our Kubeflow Pipeline for handling what will be user inputted videos, and generating corresponding key-word highlighted texts along with generated quizzes. The pipeline consists of four main components; the structure can be visualized below: 

![kubeflow image](images/kubeflow_pipeline.png)

(1) Data Conversion: This component and subdirectory found in 'src/pipeline-workflow/data-conversion' simply converts user video files (mp4) to audio files (mp3). Note that this container makes use of Dask to parallelize the process of video conversion with the moviepy library.

(2) Audio Transcription: This component and subdirectory found in 'src/pipeline-workflow/audio-transcription' transcribes the audio files to text files using the Whisper Jax model.

(3a) Keyword Extraction: This component and subdirectory found in 'src/pipeline-workflow/model-deployment' deploys our trained key-word highlighting Distilbert model and conducts inference on the transcribed text. The model is downloaded from Weights & Biases with its trained parameter set, deployed to GCP, and then used for keyword extraction.

(3b) Quiz Generation: This componet and subdirectory found in 'src/pipeline-workflow/quiz-generation' utilizes the transcribed text and forms an prompt to be inputted into the OpenAI GPT API, generating a quiz based on the lecture material.


A brief description of the pipeline usage follows proceeds:

(1) To run the complete pipeline - converting the data, transcribing the audio, extracting keywords, and generating the quiz - first enter src/pipeline-workflow/workflow. Then run 'python cli.py -p'.

(2) To run each component of the pipeline individually first enter src/pipeline-workflow/workflow. Then, for

- converting the data, run 'python cli.py -c' 
    
- transcribing the audio, run 'python cli.py -t'
    
- extracting keywords, run 'python cli.py -e'
    
- generating the quiz, run 'python cli.py -g'.


**Addenda to Presentation (10/24)**
We have included an updated set of slides (see `reports/milestone4_presentation.pdf`) with two slight adjustments:
1. The diagram on slide 3 now shows that our Quiz Generation container for now only generates quizzes based on the lecture transcript and does not yet use the keywords. A next step is to improve the keyword extraction container using prompt engineering that makes use of the keywords from the Keyword Extraction container.
2. The diagram on slide 8 now shows the Model Training container as separate from the deployed pipeline, for the sake of clarity.
