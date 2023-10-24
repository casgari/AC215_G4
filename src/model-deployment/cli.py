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

# # W&B
import wandb

GCP_PROJECT = os.environ["GCP_PROJECT"]
GCS_MODELS_BUCKET_NAME = os.environ["GCS_MODELS_BUCKET_NAME"]
BEST_MODEL = "distilbert" # ACTION: Model Name
ARTIFACT_URI = f"gs://{GCS_MODELS_BUCKET_NAME}/{BEST_MODEL}"

# ACTION: Modify details to fit model
data_details = {
    "image_width": 224,
    "image_height": 224,
    "num_channels": 3,
    "num_classes": 3,
    "labels": ["oyster", "crimini", "amanita"],
    "label2index": {"oyster": 0, "crimini": 1, "amanita": 2},
    "index2label": {0: "oyster", 1: "crimini", 2: "amanita"},
}


def download_file(packet_url, base_path="", extract=False, headers=None):
    if base_path != "":
        if not os.path.exists(base_path):
            os.mkdir(base_path)
    packet_file = os.path.basename(packet_url)
    with requests.get(packet_url, stream=True, headers=headers) as r:
        r.raise_for_status()
        with open(os.path.join(base_path, packet_file), "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    if extract:
        if packet_file.endswith(".zip"):
            with zipfile.ZipFile(os.path.join(base_path, packet_file)) as zfile:
                zfile.extractall(base_path)
        else:
            packet_name = packet_file.split(".")[0]
            with tarfile.open(os.path.join(base_path, packet_file)) as tfile:
                tfile.extractall(base_path)


def main(args=None):
    if args.upload:
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
        # download_file(
        #     "https://github.com/dlops-io/models/releases/download/v2.0/model-mobilenetv2_train_base_True.v74.zip",
        #     base_path="artifacts",
        #     extract=True,
        # )
        # artifact_dir = "./artifacts/model-mobilenetv2_train_base_True:v74/1"

        # Load model
        artifact = run.use_artifact('ac215-ppp/ppp-keyword-extraction/model-distilroberta-base-21oct:v4', type="model")
        artifact_dir = artifact.download()
        prediction_model = tf.saved_model.load(artifact_dir)

        # Save updated model to GCS
        tf.saved_model.save(
            prediction_model,
            ARTIFACT_URI
        )
        # Save updated model locally for endpoint
        # tf.saved_model.save(
        #     prediction_model,
        #     "./artifacts/endpoint/1",
        #     signatures={"serving_default": serving_function},
        # )

    elif args.deploy:
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

    elif args.predict:
        print("Predict using endpoint")

        # Get the endpoint
        # Endpoint format: endpoint_name="projects/{PROJECT_NUMBER}/locations/us-central1/endpoints/{ENDPOINT_ID}"
        endpoint = aiplatform.Endpoint(
            "projects/36357732856/locations/us-central1/endpoints/1708924743563870208"
        )

        # with open("cs109_lecture1.txt", "r") as f:
        #     sentence = f.read()
        sentence = "Diffusion models are a class of machine learning models that have gained significant attention in recent times for their ability to generate high-quality samples from complex data distributions. At their core, diffusion models operate by simulating a random process known as diffusion, which can be thought of as the reverse of a denoising process. Starting with a random noise sample, the model iteratively refines the sample through a series of steps, gradually morphing the noise into a sample that closely resembles the target data distribution. This is achieved by leveraging a sequence of transition probabilities, which guide the diffusion process. One of the primary advantages of diffusion models is their ability to generate samples without the need for explicit likelihood computation, which is a common requirement in many generative models. Their flexibility and robustness have led to applications in a variety of domains, from image synthesis to audio generation. As research progresses, the potential of diffusion models continues to unfold, promising even more advanced and diverse applications in the future."
        print(sentence)
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

    elif args.local:

        # Get a sample image to predict
        image_files = glob(os.path.join("data", "*.jpg"))
        print("image_files:", image_files[:5])
        image_samples = np.random.randint(0, high=len(image_files) - 1, size=1)
        for img_idx in image_samples:
            print("Image:", image_files[img_idx])

            with open(image_files[img_idx], "rb") as f:
                data = f.read()
            b64str = base64.b64encode(data).decode("utf-8")
            # The format of each instance should conform to the deployed model's prediction input schema.
            instances = [{"bytes_inputs": {"b64": b64str}}]

            headers = {"content-type": "application/json"}
            response = requests.post(
                "http://host.docker.internal:8501/v1/models/model_name:predict", data=data, headers=headers
            )
            print(response)
            result = json.loads(response.text)

            print("Result:", result)
            prediction = result.predictions[0]
            print(prediction, prediction.index(max(prediction)))
            print(
                "Label:   ",
                data_details["index2label"][prediction.index(max(prediction))],
                "\n",
            )



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
        "-l",
        "--local",
        action="store_true",
        help="Make prediction using endpoint locally",
    )

    args = parser.parse_args()

    main(args)
