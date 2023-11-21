from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
import asyncio
from api.tracker import TrackerService
import pandas as pd
import os
from fastapi import File
from tempfile import TemporaryDirectory
from api import model
import requests 
import random

from google.cloud import storage
import shutil

GCS_BUCKET_NAME = "mega-ppp-ml-workflow"

# Initialize Tracker Service
tracker_service = TrackerService()

# Setup FastAPI app
app = FastAPI(title="API Server", description="API Server", version="v1")

# Enable CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    print("Startup tasks")
    # Start the tracker service
    asyncio.create_task(tracker_service.track())


# Routes
@app.get("/")
async def get_index():
    return {"message": "Welcome to the API Service"}


@app.get("/experiments")
def experiments_fetch():
    # Fetch experiments
    df = pd.read_csv("/persistent/experiments/experiments.csv")

    df["id"] = df.index
    df = df.fillna("")

    return df.to_dict("records")


@app.get("/best_model")
async def get_best_model():
    model.check_model_change()
    if model.best_model is None:
        return {"message": "No model available to serve"}
    else:
        return {
            "message": "Current model being served:" + model.best_model["model_name"],
            "model_details": model.best_model,
        }


@app.post("/predict")
async def predict(file: bytes = File(...)):
    print("video file:", len(file), type(file))

    # Save the video
    num = random.randint(1, 999999)
    filename = f"video{num}"
    with TemporaryDirectory() as video_dir:
        video_path = os.path.join(video_dir, filename)
        with open(video_path, "wb") as output:
            output.write(file)
        print("")
        print(video_path)
        print("")
        # Upload video to GCP
        upload_flag = model.upload(video_path, num)
        if upload_flag:
            raise Exception("Failed to upload video")
    
        # Convert to audio using cloud function
        response = requests.get(f"https://us-central1-ac215-group-4.cloudfunctions.net/data-preprocessing?filename={filename}.mp4")
        # Transcribe with Whisper
        response = requests.get(f"https://audio-transcription-hcsan6rz2q-uc.a.run.app/?filename={filename}.mp3")
       
    
    transcript_path = download("text_prompts", f"{filename}.txt")


    # Extract keywords using endpoint
    prediction_results = {}
    prediction_results = model.make_prediction_vertexai(transcript_path)

    # TODO: ADD KEYWORDS TO TRANSCRIPT BEFORE GENERATING QUIZ

     # Generate quiz using cloud function
    response = requests.get(f"https://us-central1-ac215-group-4.cloudfunctions.net/quiz-generation?filename={filename}.txt")
    quiz = response.text
    print("DONE!!!!")

    # edit return results
    prediction_results["quiz"] = quiz
    print(prediction_results)
    return prediction_results

def download(folder, filename):
    print("download")

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(GCS_BUCKET_NAME)
    # bucket = storage_client.get_bucket(bucket_n) 

    # Clear
    shutil.rmtree(folder, ignore_errors=True, onerror=None)
    os.makedirs(folder)

    blobs = bucket.list_blobs(prefix=folder + "/")
    for blob in blobs:
        if blob.name == (folder + "/" + filename):
            blob.download_to_filename(blob.name)
            return blob.name