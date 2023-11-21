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
    │   ├── application_design.md
    │   ├── milestone2.md
    │   ├── milestone3.md
    │   ├── milestone4.md
    │   └── milestone4_presentation.md
    ├── images               <- Folder containing images used in reports
    ├── cli.py               <- Files in the root of repo for data versioning
    ├── docker-shell.sh
    ├── Dockerfile
    ├── Pipfile
    ├── Pipfile.lock
    ├── requirements.txt
    ├── keyword_dataset.dvc
    └── src                  <- Source code and Dockerfiles for training and deployment
        ├── individual-containers   <- initial iteration of containers to be run independently of the pipeline, from earlier milestones
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
        │   └── ...
        └── deployment     <- Code for deployment to GCP with Ansible
            └── ...


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

Finally, we used Ansible to create, provision, and deploy our frontend and backend to GCP in an automated fashion.

In the remainder of this update, we will explain the code structure of the [deliverables](https://harvard-iacs.github.io/2023-AC215/milestone5/) for Milestone 5. Given the size of the codebase, please note that we reserve this space for describing the deliverables for Milestone 5 only, and usage of other parts of the codebase is desribed in earlier reports (see `reports/`).

#### Code Structure

**Backend API Service Implementation**

TODO

**Frontend Implementation**

TODO

**Ansible Usage**

TODO