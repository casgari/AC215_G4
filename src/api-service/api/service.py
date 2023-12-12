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
import shutil

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
        print("Start Filetype Conversion")
        response = requests.get(f"https://us-central1-ac215-group-4.cloudfunctions.net/data-preprocessing?filename={filename}.mp4")
        # Transcribe with Whisper
        print("Start Audio Transcription")
        response = requests.get(f"https://audio-transcription-hcsan6rz2q-uc.a.run.app/?filename={filename}.mp3")
    
    print("Start Transcript Analysis")  
    transcript_path = model.download("text_prompts", f"{filename}.txt")

    # Extract keywords using endpoint
    prediction_results = {}
    prediction_results = model.make_prediction_vertexai(transcript_path)

    # TODO: ADD KEYWORDS TO TRANSCRIPT BEFORE GENERATING QUIZ

     # Generate quiz using cloud function
    print("Start Quiz Generation")
    response = requests.get(f"https://us-central1-ac215-group-4.cloudfunctions.net/quiz-generation?filename={filename}.txt")
    quiz = response.text
    print("DONE!!!!")

    # edit return results
    prediction_results["quiz"] = quiz
    print(prediction_results)
    return prediction_results


@app.post("/predicttext")
async def predict_text(file: bytes = File(...)):
    print("text file:", len(file), type(file))

    # Save the video
    num = random.randint(1, 999999)
    filename = f"video{num}"
    with TemporaryDirectory() as text_dir:
        text_path = os.path.join(text_dir, filename)
        with open(text_path, "wb") as output:
            output.write(file)
        print("")
        print(text_path)
        print("")
        # Upload video to GCP
        upload_flag = model.upload_text(text_path, num)
        if upload_flag:
            raise Exception("Failed to upload text file")
       
    print("Start Transcript Analysis")
    print(filename)
    transcript_path = model.download("text_prompts", f"{filename}.txt")

    # Extract keywords using endpoint
    prediction_results = {}
    prediction_results = model.make_prediction_vertexai(transcript_path)
    local_kw_file = ','.join(prediction_results['prediction_label'])
    kw_file_path = f"keywords{num}.txt"

    # Write the contents to kw_file_path
    with open(kw_file_path, 'w') as file:
        file.write(local_kw_file)

    filename_kw = f"keywords{num}"
    with TemporaryDirectory() as text_dir:
        # Copy the file to the temporary directory
        temp_file_path = os.path.join(text_dir, kw_file_path)
        shutil.copy(kw_file_path, temp_file_path)

        # Upload video to GCP
        upload_flag = model.upload_kw(temp_file_path, num)
        if upload_flag:
            raise Exception("Failed to upload text file")

    # TODO: ADD KEYWORDS TO TRANSCRIPT BEFORE GENERATING QUIZ
    
     # Generate quiz using cloud function
    response = requests.get(f"https://us-central1-ac215-group-4.cloudfunctions.net/quiz-generation?filename={filename}.txt")
    quiz = response.text
    print("DONE!!!!")

    response = requests.get(f"https://us-central1-ac215-group-4.cloudfunctions.net/clean-keywords?filename={filename}.txt&keywords={filename_kw}.txt")
    cleaned_kw = response.text
    # edit return results
    prediction_results["quiz"] = quiz
    prediction_results["keywords"] = cleaned_kw
    print(prediction_results)
    return prediction_results 