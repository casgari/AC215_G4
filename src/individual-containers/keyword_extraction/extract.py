"""
Module that contains the command line app.
"""
import os
import io
import argparse
import shutil
from google.cloud import storage
from keybert import KeyBERT


# Generate the inputs arguments parser
parser = argparse.ArgumentParser(description="Command description.")

gcp_project = "AC215 Group 4"
bucket_name = "mega-ppp"
input_transcripts = "text_prompts"
keywords = "keywords"


def makedirs():
    os.makedirs(input_transcripts, exist_ok=True)
    os.makedirs(keywords, exist_ok=True)


def download():
    print("download")

    # Clear
    shutil.rmtree(input_transcripts, ignore_errors=True, onerror=None)
    makedirs()

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=input_transcripts + "/")
    for blob in blobs:
        print(blob.name)
        if not blob.name.endswith("/"):
            blob.download_to_filename(blob.name)


def extract():
    print("extract")
    kw_model = KeyBERT() # adjust the BERT model here using the `model` argument
    makedirs()

    # Get the list of audio file
    text_files = os.listdir(input_transcripts)

    for text_path in text_files:
        uuid = text_path.replace(".txt", "")
        text_path = os.path.join(input_transcripts, text_path)
        text_file = os.path.join(keywords, uuid + ".txt")

        if os.path.exists(text_file):
            continue

        print("Extracting:", text_path)

        with open(text_path) as f:
            this_file = f.read()
            
            # Extract top 10 keywords
            keywords_list = kw_model.extract_keywords(this_file,
                                                      keyphrase_ngram_range=(1, 3),
                                                      top_n=10)
            
            #convert key word list to str of keywords to output
            keywords_str = ''.join([''.join(str(x[0])+'\n') for x in keywords_list])

            # Save the transcription
            with open(text_file, "w") as f:
                f.write(keywords_str)


def upload():
    print("upload")
    makedirs()

    # Upload to bucket
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Get the list of text file
    text_files = os.listdir(keywords)

    for text_file in text_files:
        file_path = os.path.join(keywords, text_file)

        destination_blob_name = file_path
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(file_path)


def main(args=None):
    print("Args:", args)

    if args.download:
        download()
    if args.extract:
        extract()
    if args.upload:
        upload()


if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal 'python cli.py --help', it will provide the description
    parser = argparse.ArgumentParser(description="Extract keywords from text files")

    parser.add_argument(
        "-d",
        "--download",
        action="store_true",
        help="Download audio files from GCS bucket",
    )

    parser.add_argument(
        "-e", "--extract", action="store_true", help="Extract keywords from text"
    )

    parser.add_argument(
        "-u",
        "--upload",
        action="store_true",
        help="Upload keywords as .txt file to GCS bucket",
    )

    args = parser.parse_args()

    main(args)
