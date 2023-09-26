AC215-Template (Milestone2)
==============================

AC215 - Milestone2

Project Organization
------------
      ├── LICENSE
      ├── README.md
      ├── Dockerfile
      ├── cli.py
      ├── notebooks
      ├── references
      ├── requirements.txt
      ├── setup.py
      └── src
            ├── generate_quiz
            │   ├── Dockerfile
            │   └── generate.py
            └── keyword_extraction
                  ├── Dockerfile
                  └── extract.py


--------
# AC215 - Milestone2 - ButterFlyer

**Team Members**
Cyrus Asgari, Ben Ray, Caleb Saul, Warren Sunada Wong, Chase Van Amburg

**Group Name**
Pavlos' Perceptron Pals

**Project**
In this project we aim to develop an application that can process lecture videos to generate transcripts with key-word highlighting and offer auto-generated question and answer flashcards.

### Milestone2 ###

We added a toy example video that we recorded ourselves, in addition to the [Inspec](https://huggingface.co/datasets/midas/inspec) dataset of keywords from abstracts. 

**Preprocessing container**
- This container takes mp4 video files, converts them to mp3, and stores it back to GCP
- Input to this container is source and destination GCS location, secrets needed - via docker
- Output from this container stored at GCS location

(1) `src/preprocess_audio/convert.py`  - Here we do preprocessing of video files, converting them to mp3.

(2) `src/preprocess_audio/Pipfile` - We used following packages to help us preprocess here - `moviepy, ffmpeg` 

(3) `src/preprocess_audio/Dockerfile` - This dockerfile starts with  `python:3.8-slim-buster`. This <statement> attaches volume to the docker container and also uses secrets (not to be stored on GitHub) to connect to GCS.

To run Dockerfile - `Run ./docker-shell.sh`

**Transcription container**
- This container takes mp3 files and applies the JAX-whisper model to create a .txt transcript of the audio file, and stores it back to GCP.
- Input to this container is source and destination GCS location, secrets needed - via docker
- Output from this container stored at GCS location

(1) `src/transcribe_audio/transcribe.py`  - Here we do transcription of audio files, outputting txt files.

(2) `src/transcribe_audio/Pipfile` - We used following packages to help us here - `whisper-jax, jax, jaxlib, ffmpeg` 

(3) `src/transcribe_audio/Dockerfile` - This dockerfile starts with  `python:3.8-slim-buster`. This <statement> attaches volume to the docker container and also uses secrets (not to be stored on GitHub) to connect to GCS.

To run Dockerfile - `Run ./docker-shell.sh`

**Keyword container**
- This container takes .txt transcript files and applies KeyBERT to create a list of keywords, and stores it back to GCP.
- Input to this container is source and destination GCS location, secrets needed - via docker
- Output from this container stored at GCS location

(1) `src/keyword_extraction/extract.py`  - Here we do transcription of audio files, outputting txt files.

(2) `src/keyword_extraction/Pipfile` - We used following packages to help us here - `keybert, scikit-learn` 

(3) `src/keyword_extraction/Dockerfile` - This dockerfile starts with  `python:3.8-slim-buster`. This <statement> attaches volume to the docker container and also uses secrets (not to be stored on GitHub) to connect to GCS.

To run Dockerfile - `Run ./docker-shell.sh`

**Generate Quiz**
- This container takes .txt transcript files and applies the OpenAI API to create a series of multiple choice comprehension questions, and stores it back to GCP.
- Input to this container is source and destination GCS location, secrets needed - via docker
- Output from this container stored at GCS location

(1) `src/keyword_extraction/generate.py`  - Here we call the OpenAI API to create the quizzes.

(2) `src/keyword_extraction/Pipfile` - We used following packages to help us here - `openai` 

(3) `src/keyword_extraction/Dockerfile` - This dockerfile starts with  `python:3.8-slim-buster`. This <statement> attaches volume to the docker container and also uses secrets (not to be stored on GitHub) to connect to GCS.

To run Dockerfile - `Run ./docker-shell.sh`


**Data Versioning**
- This container (in the root directory) applies data versioning to the keyword_dataset folder.
- Input to this container is source GCS location, secrets needed - via docker
- Output is data versioned files
  
(1) `cli.py` - File used to download the data from the GCS bucket. 

(2) `Pipfile` - We used following packages to help us here - `dvc, dvc-gs` 

(3) `src/validation/Dockerfile` - This dockerfile starts with  `python:3.8-slim-buster`. This <statement> attaches volume to the docker container and also uses secrets (not to be stored on GitHub) to connect to GCS.

To run Dockerfile - `Run ./docker-shell.sh`


----
