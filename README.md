AC215 - Milestone3

Project Organization
------------

    .
    ├── data # DO NOT UPLOAD DATA
    │   ├── interim          <- Intermediate preprocessed data
    │   │   ├── test.csv
    │   │   ├── train.csv
    │   │   └── val.csv
    │   ├── processed        <- Final dataset files for modeling
    │   │   ├── file_00-0.tfrec
    │   │   ├── file_00-1.tfrec
    │   │   ├── file_00-2.tfrec
    │   │   └── file_00-3.tfrec
    │   └── raw              <- Original immutable input data
    │       └── training_data.zip
    ├── LICENSE
    ├── notebooks            <- Jupyter notebooks for EDA and model testing
    │   ├── intial_model_construction.ipynb
    │   ├── tf_intial_model_construction_with_multigpu.ipynb
    │   ├── tf_dask_test.ipynb
    │   └── distributed_training_demo.ipynb
    ├── README.md
    ├── references           <- Reference materials such as papers
    ├── reports              <- Folder containing your milestone markdown submissions
    │   ├── milestone2.md
    │   └── milestone3.md
    ├── requirements.txt
    ├── Dockerfile
    ├── cli.py
    ├── setup.py
    ├── .dvc      
    └── src                  <- Source code and Dockerfiles for data processing and modeling
        ├── data    <- Scripts for dataset creation
        │   ├── build_records.py
        │   ├── dataloader.py
        │   ├── Dockerfile
        │   ├── process.py
        ├── dataloader    <- Scripts for dataset creation
        │   ├── tokenizer.py
        │   ├── Dockerfile
        │   ├── docker-shell.sh
        │   ├── Pipfile.lock
        │   └── Pipfile
        ├── preprocess_audio_file
        │   ├── Dockerfile
        │   ├── Pipfile
        │   ├── Pipfile.lock
        │   ├── docker-shell.sh
        │   └── convert.py
        ├── transcribe_audio
        │   ├── Dockerfile
        │   ├── Pipfile
        │   ├── Pipfile.lock
        │   ├── docker-shell.sh
        │   └── transcribe.py
        ├── generate_quiz
        │   ├── Dockerfile
        │   ├── Pipfile
        │   ├── Pipfile.lock
        │   ├── docker-shell.sh
        │   └── generate.py
        ├── keyword_extraction
        │   ├── Dockerfile
        │   ├── Pipfile
        │   ├── Pipfile.lock
        │   ├── docker-shell.sh
        │   └── extract.py
        ├── model_training
        │   ├── Dockerfile
        │   ├── Pipfile
        │   ├── Pipfile.lock
        │   ├── docker-shell.sh
        │   └── trainer.py
        └── vertex_training
            ├── cli.py
            ├── cli.sh
            ├── Pipfile.lock
            ├── Pipfile
            ├── Dockerfile
            ├── docker-shell.sh
            ├── package-trainer.sh
            └── package
                ├── PKG-INFO
                ├── setup.cfg
                ├── setup.py
                └── trainer
                    ├── task_multigpu.py
                    ├── task_reference.py
                    └── __init__.py
                

--------
# AC215 - Milestone3 - Learning Tools for Transcribed Lecture Audio

**Team Members**
Cyrus Asgari, Ben Ray, Caleb Saul, Warren Sunada Wong, Chase Van Amburg

**Group Name**
Pavlos' Perceptron Pals

**Project**
In this project we aim to develop an application that can process lecture videos to generate transcripts with key-word highlighting and offer auto-generated quizzes with both questions and answers.

**Recap of work to date (see `reports/milestone2.md` for more details)**
We have built the four main containers that we will use in our pipeline for deployment, in addition to a container that takes care of data versioning. On GCP, we have a toy example video that we recorded ourselves, in addition to the [Inspec](https://huggingface.co/datasets/midas/inspec) dataset of keywords from abstracts, which we will use for training the keyword extraction model in the next milestone. With our toy example, we were able to verify that the four atomic containers input and output what is expected, albeit further optimization is expected as we progress.

### Milestone3 ###

Our efforts in this milestone focused on building out our pipeline for training our keyword extraction model. This includes the creation of two new containers - one for preprocessing and one for running training.

Our initial experimentation led us to build a naive preprocessing and training script (`notebooks/initial_model_construction.ipynb`) using Pytorch data types on a single GPU, and we iterated from there. We now use TensorFlow Data throughout preprocessing and training, implement Dask for efficient transformations to our dataset, and add support for training on multiple GPUs - all while tracking performance with Weights & Biases.

In the preprocessing stage, we apply a tokenizer to the keywords in the [Inspec](https://huggingface.co/datasets/midas/inspec) dataset of keywords, using the TF Dataset of tokenized key words (known as `input_ids`) and labels (`B`, `I`, or `O`, depending on whether beginning, inside, or outside a keyword phrase) in training.

In training, we have begun experimenting with different versions of the BERT model (accessed via `TFAutoModelForTokenClassification`) to improve our performance on keyword extraction. Although Google has not yet approved our request for muliple GPU compute instances, we have added support for training on multiple GPUs, using TensorFlow's support for distributed training (i.e. `tf.distribute.MirroredStrategy()`). We have tracked our performance of the several training runs completed to date using Weights & Biases, but will be doing more experimentation and optimization (including a hyperparameter search) for Milestone4.

In the remainder of this update, we will explain the code structure of the three [deliverables](https://harvard-iacs.github.io/2023-AC215/milestone3/#deliverables) for Milestone3.

#### Code Structure

**Data Pipeline Implementation**
As a reminder from milestone2, our data versioning container sits in the root of our GitHub repository, applying DVC to the `keyword_dataset` folder (which is not pushed to GitHub). The input to this container is the source GCS location, accessible via the `secrets` folder through Docker. The output of this container is the data versioning files.
  
We have now added a second container for preprocessing our [Inspec](https://huggingface.co/datasets/midas/inspec) dataset of keywords for training. This can be found in `src/dataloader` with usage as follows:

(1) `src/dataloader/tokenizer.py`  - This script loads the [Inspec](https://huggingface.co/datasets/midas/inspec) data to our compute instance's local `keyword_dataset` folder for tokenization.

(2) `src/dataloader/Pipfile` - We used following packages to help us here - `transformers, google-cloud-storage, datasets, gcsfs, numpy` 

(3) `src/dataloader/Dockerfile` - This Dockerfile starts with `python:3.8-slim-buster`. This <statement> attaches volume to the Docker container and also uses secrets (not to be stored on GitHub) to connect to GCS. To run Dockerfile - `sh docker-shell.sh`


**Distributed Computing and Storage Integration**
We have implemented support for Dask in `notebooks/tf_dask_test`, in case we wish to make additional transformations to our dataset in furture. We have not had to perform any dataset-wide transformations beyond tokenization, which uses built-in TensorFlow functions rather than Dask, so we have not shifted this over to the preprocessing container for now.

We take further advantage of distributed computing by handling our data as TF Datasets, shuffling the data appropriately, and prefetching batch(es) of the data for training (on multiple GPUs if available). Look for the `model.prepare_tf_dataset` method in the training container (see `src/model_training/trainer.py`) for our usage of TF Datasets. The training container is described further below.

**Machine Learning Workflow Implementation**

Having completed preprocessing, we train our model using the `src/model_training` container with usage as follows:

(1) `src/model_training/trainer.py`  - this script loads the preprocessed training and validation data from GCS, runs the training loop, and tracks the performance via Weights & Biases.

(2) `src/model_training/Pipfile` - We used following packages to help us here - `transformers, datasets, seqeval, evaluate, wandb, tensorflow` 

(3) `src/Model_training/Dockerfile` - This Dockerfile starts with `python:3.8-slim-buster`. This <statement> attaches volume to the Docker container and also uses secrets (not to be stored on GitHub) to connect to GCS. To run Dockerfile - `sh docker-shell.sh`


Below you can see the output from our Weights & Biases page. We used this tool to track several iterations of our model training. It was tracked using the `wandb` library we included inside of our `trainer.py` script. 

![wnb image](images/wandb.png)


**Vertex AI Integration**

Finally, we integrated our entire training process with Vertex AI. This allows us to run multiple jobs (serverless training) with multi-GPU support, instead of training on local machines or on Colab. This still allows for Weights and Biases reporting during model training. The model is conveniently stored in a bucket on the cloud, where it can later be accessed for deployment. 

(1) `src/vertex_training/docker-shell.sh` creates a container that sets up our training method to integrate with Google Cloud storage and Vertex AI.

(2) `src/vertex_training/package-trainer.sh` creates a tar.gz file that bundles all training code and uploads it to the `model-trainer` bucket.

(3) `src/vertex_training/cli.sh` provides args to the `task_multigpu.py` file that trains the provided model with capacity to train on multiple GPUs. `cli.sh` also launches this training job on custom training GPUs accessed through Vertex AI. Trained model is stored on the cloud in a `model-trainer` bucket.
