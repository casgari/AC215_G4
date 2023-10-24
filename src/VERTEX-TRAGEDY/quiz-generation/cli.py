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

gcp_project = "ac215-group-4"
bucket_name = "mega-ppp-ml-workflow"
text_prompts = "text_prompts"
generated_quizzes = "generated_quizzes"
# openai_key = "../../../secrets/openai_api_key.txt"

def makedirs():
    os.makedirs(text_prompts, exist_ok=True)
    os.makedirs(generated_quizzes, exist_ok=True)

def download():
    print("download")

    # Clear
    shutil.rmtree(text_prompts, ignore_errors=True, onerror=None)
    makedirs()

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=text_prompts + "/")
    for blob in blobs:
        print(blob.name)
        if not blob.name.endswith("/"):
            blob.download_to_filename(blob.name)

def generate():
    print("generate")
    text_files = os.listdir(text_prompts)

    #THINK I NEED SECRETS HERE TO BE ABLE TO TEST THIS

    # with open(openai_key) as f:
    #     openai.api_key = f.read()
    hidden_key = "k-fMArUy3cgKt9Bb242NWuT3BlbkFJo8kpsYY7bE74KE17Sv30" 
    openai.api_key = "s" + hidden_key

    for text_file in text_files:
        uuid = text_file.replace(".txt", "")
        file_path = os.path.join(text_prompts, text_file)
        quiz_path = os.path.join(generated_quizzes, uuid + "_quiz.txt")

        with open(file_path, "r") as f:
            lecture_transcript = f.read()
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"Please generate 5 multiple choice questions for the following lecture: {lecture_transcript[:3800]}"}
            ])
        quiz = response.choices[0].message.content
        with open(quiz_path, "w") as f:
            f.write(quiz)

def upload():
    print("upload")
    makedirs()

    # Upload to bucket
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Get the list of text file
    quiz_files = os.listdir(generated_quizzes)

    for quiz_file in quiz_files:
        file_path = os.path.join(generated_quizzes, quiz_file)

        destination_blob_name = file_path
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(file_path)

def main(args=None):
    print("Args:", args)

    if args.generate:
        download()
        generate()
        upload()


if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal 'python cli.py --help', it will provide the description
    parser = argparse.ArgumentParser(description="Transcribe audio file to text")

    parser.add_argument(
        "-g", "--generate", action="store_true", help="Generate quizzes from transcript"
    )

    args = parser.parse_args()

    main(args)
