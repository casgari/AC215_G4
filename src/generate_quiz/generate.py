"""
Module that contains the command line app.
"""
import os
import argparse
import shutil
from google.cloud import storage
import openai

# Generate the inputs arguments parser
parser = argparse.ArgumentParser(description="Command description.")

gcp_project = "AC215 Group 4"
bucket_name = "mega-ppp"
input_prompts = "text_prompts"
generated_quizzes = "generated_quizzes"

def makedirs():
    os.makedirs(input_prompts, exist_ok=True)
    os.makedirs(generated_quizzes, exist_ok=True)

def download():
    print("download")

    # Clear
    shutil.rmtree(input_prompts, ignore_errors=True, onerror=None)
    makedirs()

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=input_prompts + "/")
    for blob in blobs:
        print(blob.name)
        if not blob.name.endswith("/"):
            blob.download_to_filename(blob.name)

def generate():
    print("generate")

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

def main(args=None):
    print("Args:", args)

    if args.download:
        download()
    if args.generate:
        generate()
    if args.upload:
        upload()


if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal 'python cli.py --help', it will provide the description
    parser = argparse.ArgumentParser(description="Transcribe audio file to text")

    parser.add_argument(
        "-d",
        "--download",
        action="store_true",
        help="Download audio files from GCS bucket",
    )

    parser.add_argument(
        "-g", "--generate", action="store_true", help="Generate quizzes from transcript"
    )

    parser.add_argument(
        "-u",
        "--upload",
        action="store_true",
        help="Upload quizzes to GCS bucket",
    )

    args = parser.parse_args()

    main(args)
