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
    with TemporaryDirectory() as video_dir:
        video_path = os.path.join(video_dir, f"video{num}.mp4")
        with open(video_path, "wb") as output:
            output.write(file)
        
        # Upload video to GCP
        uploaded = model.upload(video_path)
        if not uploaded:
            raise Exception("Failed to upload video")
        print("DONE!!!!!")
        exit()
    
    # Convert to audio using cloud function
    # response = requests.get("https://us-central1-ac215-group-4.cloudfunctions.net/data-preprocessing?"
    # Transcribe audio file using cloud run

    # Extract keywords using endpoint
    
    # Generate quiz using cloud function

    # Make prediction
    prediction_results = {}
    prediction_results = model.make_prediction_vertexai(image_path)

    print(prediction_results)
    return prediction_results
