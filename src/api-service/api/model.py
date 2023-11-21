import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.keras.models import Model
import tensorflow_hub as hub
from google.cloud import aiplatform, storage
import base64
import shutil
import transformers
from transformers import AutoTokenizer


GCS_BUCKET_NAME = "mega-ppp-ml-workflow"

def upload(path, num):
    gcp_project = "ac215-group-4"
    filename = f"video{num}.mp4"

    # Upload to bucket
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    destination_blob_name = f"input_videos/{filename}"
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(path)
    return 0

def upload_text(path, num):
    gcp_project = "ac215-group-4"
    filename = f"text{num}.mp4"

    # Upload to bucket
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    destination_blob_name = f"text_prompts/{filename}"
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(path)
    return 0

def download(folder, filename):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(GCS_BUCKET_NAME)
    # bucket = storage_client.get_bucket(bucket_n) 

    # Clear
    shutil.rmtree(folder, ignore_errors=True, onerror=None)
    os.makedirs(folder)

    blobs = bucket.list_blobs(prefix=folder + "/")
    for blob in blobs:
        blob_name = blob.name[:-4] + ".txt"
        if blob_name == (folder + "/" + filename):
            blob.download_to_filename(blob_name)
            return blob_name


def make_prediction_vertexai(image_path):
    print("Predict using Vertex AI endpoint")

    # Get the endpoint
    # Endpoint format: endpoint_name="projects/{PROJECT_NUMBER}/locations/us-central1/endpoints/{ENDPOINT_ID}"
    endpoint = aiplatform.Endpoint(
        "projects/36357732856/locations/us-central1/endpoints/1708924743563870208"
    )

    with open(image_path, "rb") as f:
        data = f.read()
    data = data.decode("utf-8")
    
    # tokenize the data
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    example_text = tokenizer(data, truncation=True, return_token_type_ids=True)

    # The format of each instance should conform to the deployed model's prediction input schema.
    instances = [example_text]

    result = endpoint.predict(instances=instances)

    print("Result:", result)
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
    keyphrases_string = ", ".join(keyphrases)

    return {
        "prediction_label": keyphrases_string,
    }