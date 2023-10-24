"""
Module that contains the command line app.

Typical usage example from command line:
        python cli.py --upload
        python cli.py --deploy
        python cli.py --predict
"""

import os
import requests
import json
import zipfile
import tarfile
import argparse
from glob import glob
import numpy as np
import base64
from google.cloud import storage
from google.cloud import aiplatform
import tensorflow as tf


import transformers
from transformers import TFAutoModelForTokenClassification
from transformers import AutoTokenizer
import shutil

# # W&B
import wandb

GCP_PROJECT = "ac215-group-4"
GCS_MODELS_BUCKET_NAME = "mega-ppp-ml-workflow"
BEST_MODEL = "distilbert" # ACTION: Model Name
ARTIFACT_URI = f"gs://{GCS_MODELS_BUCKET_NAME}/{BEST_MODEL}"

text_prompts = "text_prompts"

def makedirs():
    os.makedirs(text_prompts, exist_ok=True)

def download():
    print("download")

    # Clear
    shutil.rmtree(text_prompts, ignore_errors=True, onerror=None)
    makedirs()

    client = storage.Client()
    bucket = client.get_bucket(GCS_MODELS_BUCKET_NAME)

    blobs = bucket.list_blobs(prefix=text_prompts + "/")
    for blob in blobs:
        print(blob.name)
        if not blob.name.endswith("/"):
            blob.download_to_filename(blob.name)

def upload():
    print("Upload model to GCS")

    storage_client = storage.Client(project=GCP_PROJECT)
    bucket = storage_client.get_bucket(GCS_MODELS_BUCKET_NAME)

    # ACTION: Use this code if you want to pull your model directly from WandB

    with open(os.environ["WANDB_KEY"], "r") as f:
        WANDB_KEY = f.read()
    # Login into wandb
    wandb.login(key=WANDB_KEY)

    # Download model artifact from wandb
    run = wandb.init()

    # Load model
    artifact = run.use_artifact('ac215-ppp/ppp-keyword-extraction/model-distilroberta-base-21oct:v4', type="model")
    artifact_dir = artifact.download()
    prediction_model = tf.saved_model.load(artifact_dir)

    # Save updated model to GCS
    tf.saved_model.save(
        prediction_model,
        ARTIFACT_URI
    )

def deploy():
    print("Deploy model")

    # List of prebuilt containers for prediction
    # https://cloud.google.com/vertex-ai/docs/predictions/pre-built-containers
    serving_container_image_uri = (
        "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest"
    )

    # Upload and Deploy model to Vertex AI
    # Reference: https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.Model#google_cloud_aiplatform_Model_upload
    deployed_model = aiplatform.Model.upload(
        display_name=BEST_MODEL,
        artifact_uri=ARTIFACT_URI,
        serving_container_image_uri=serving_container_image_uri,
    )
    print("deployed_model:", deployed_model)
    # Reference: https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.Model#google_cloud_aiplatform_Model_deploy
    endpoint = deployed_model.deploy(
        deployed_model_display_name=BEST_MODEL,
        traffic_split={"0": 100},
        machine_type="n1-standard-4",
        accelerator_count=0,
        min_replica_count=1,
        max_replica_count=1,
        sync=False,
    )
    print("endpoint:", endpoint)

def predict():
    print("Predict using endpoint")

    # Get the endpoint
    # Endpoint format: endpoint_name="projects/{PROJECT_NUMBER}/locations/us-central1/endpoints/{ENDPOINT_ID}"
    endpoint = aiplatform.Endpoint(
        "projects/36357732856/locations/us-central1/endpoints/1708924743563870208"
    )

    # GET LECTURE TRANSCRIPTS
    download()

    text_files = os.listdir(text_prompts)
    for text_file in text_files:
        # DOWNLOAD LECTURE TRANSCRIPTS INTO text_prompts/
        uuid = text_file.replace(".txt", "")
        file_path = os.path.join(text_prompts, text_file)
        keyword_path = os.path.join("extracted_keywords", uuid + "_keywords.txt")

        with open(file_path, "r") as f:
            sentence = f.read()
        
        tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
        example_text = tokenizer(sentence, truncation=True, return_token_type_ids=True)


        # The format of each instance should conform to the deployed model's prediction input schema.
        

        result = endpoint.predict(instances=[example_text])

        logits = result.predictions[0]
        
        predictions = np.argmax(logits, axis=-1)
        keyphrases = []
        keyphrase = []
        for label, token in zip(predictions, example_text["input_ids"]):
            if label == 0:
                keyphrase = [tokenizer.decode(token)]
            elif label == 1 and len(keyphrase) > 0:
                keyphrase.append(tokenizer.decode(token))
            elif label == 2 and len(keyphrase) > 0:
                keyphrases.append(''.join(keyphrase))
                keyphrase = []
        print("Keywords:", keyphrases)
        ## UPLOAD TO BUCKET
        keyphrases_string = "\n".join(keyphrases)

        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_MODELS_BUCKET_NAME)

        destination_blob_name = keyword_path
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_string(keyphrases_string)

def main(args=None):
    if args.upload:
        upload()
    elif args.deploy:
        deploy()

    elif args.predict:
        predict()

    elif args.generate:
        upload()
        predict()



if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal 'python cli.py --help', it will provide the description
    parser = argparse.ArgumentParser(description="Data Collector CLI")

    parser.add_argument(
        "-u",
        "--upload",
        action="store_true",
        help="Upload saved model to GCS Bucket",
    )
    parser.add_argument(
        "-d",
        "--deploy",
        action="store_true",
        help="Deploy saved model to Vertex AI",
    )
    parser.add_argument(
        "-p",
        "--predict",
        action="store_true",
        help="Make prediction using the endpoint from Vertex AI",
    )
    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        help="Test deployment to Vertex AI",
    )
    parser.add_argument(
        "-g",
        "--generate",
        action="store_true",
        help="Do all three: upload model, deploy endpoint, predict keywords",
    )

    args = parser.parse_args()

    main(args)
