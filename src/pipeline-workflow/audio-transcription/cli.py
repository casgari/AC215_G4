"""
Module that contains the command line app.
"""
import os
import argparse
import shutil
from google.cloud import storage
from jax.experimental.compilation_cache import compilation_cache as cc
from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp

# Generate the inputs arguments parser
parser = argparse.ArgumentParser(description="Command description.")

gcp_project = "ac215-group-4"
bucket_name = 'mega-ppp-ml-workflow'
input_audios = "audio_files"
text_prompts = "text_prompts"

def makedirs():
    os.makedirs(input_audios, exist_ok=True)
    os.makedirs(text_prompts, exist_ok=True)

def download(filename=None):
    print("download")

    # Clear
    shutil.rmtree(input_audios, ignore_errors=True, onerror=None)
    makedirs()

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=input_audios + "/")
    for blob in blobs:
        print(blob.name)
        if filename is None: # Download all files
            if not blob.name.endswith("/"):
                blob.download_to_filename(blob.name)
        else: # Download specific file
            if blob.name.endswith("/" + filename):
                blob.download_to_filename(blob.name)


def transcribe():
    print("transcribe")
    makedirs()
    cc.initialize_cache("./jax_cache")

    pipeline = FlaxWhisperPipline("openai/whisper-small", dtype=jnp.bfloat16, batch_size=16)

    # Get the list of audio file
    audio_files = os.listdir(input_audios)

    for audio_path in audio_files:
        uuid = audio_path.replace(".mp3", "")
        audio_path = os.path.join(input_audios, audio_path)
        text_file = os.path.join(text_prompts, uuid + ".txt")

        if os.path.exists(text_file):
            continue

        print("Transcribing:", audio_path)
        text = pipeline(audio_path)

        # Save the transcription
        with open(text_file, "w") as f:
            f.write(text['text'])

def upload():
    print("upload")
    makedirs()

    # Upload to bucket
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Get the list of text file
    text_files = os.listdir(text_prompts)

    for text_file in text_files:
        file_path = os.path.join(text_prompts, text_file)

        destination_blob_name = file_path
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(file_path)

    return text_files


def transcribe_file(filename):
    download(filename=filename)
    transcribe()
    return upload() # return a list of text files uploaded to the bucket.

def main(args=None):
    print("Args:", args)

    if args.transcribe:
        download()
        transcribe()
        upload()

    if args.filename != "":
        transcribe_file(args.filename)

if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal 'python cli.py --help', it will provide the description
    parser = argparse.ArgumentParser(description="Transcribe audio file to text")

    parser.add_argument(
        "-t", "--transcribe", action="store_true", help="Transcribe all audio files in bucket to text"
    )

    parser.add_argument(
        "-f", "--filename", type=str, default="", help="Transcribe specific audio file to text"
    )

    args = parser.parse_args()

    main(args)
