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
       
    
    transcript_path = model.download("text_prompts", f"{filename}.txt")


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
    