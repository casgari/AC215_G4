"""
Module that contains the command line app.

Typical usage example from command line:
        python cli.py --clean
"""

import os
import argparse
from glob import glob
import shutil
import uuid
import pandas as pd
import zipfile
from google.cloud import storage
import tfrecords
import cleanser


dataset_folder = os.path.join("/persistent", "dataset")
raw_folder = os.path.join(dataset_folder, "raw")
clean_folder = os.path.join(dataset_folder, "clean")
tfrecords_folder = os.path.join(dataset_folder, "tfrecords")


def main(args=None):
    GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

    # If bucket was passed as argument
    if args.bucket != "":
        GCS_BUCKET_NAME = args.bucket
    print("GCS_BUCKET_NAME:", GCS_BUCKET_NAME)

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(GCS_BUCKET_NAME)

    # Make dirs
    os.makedirs(dataset_folder, exist_ok=True)

    if args.clean:
        print("Cleaning dataset")

        # Check if raw exists
        if not os.path.exists(raw_folder):
            # Download zip file
            source_blob_name = "raw.zip"
            destination_file_name = os.path.join(dataset_folder, "raw.zip")
            blob = bucket.blob(source_blob_name)
            blob.download_to_filename(destination_file_name)

            # Unzip data
            with zipfile.ZipFile(destination_file_name) as zfile:
                zfile.extractall(raw_folder)

        # Clean data
        #############
        # Remove duplicates
        cleanser.remove_duplicates(raw_folder, clean_folder)

        # Verify image files
        cleanser.verify_images(clean_folder)

        # Zip dataset
        data_list = glob(clean_folder + "/*/*")
        zip_file = os.path.join(dataset_folder, "clean.zip")
        with zipfile.ZipFile(zip_file, "w") as zip:
            for file in data_list:
                zip.write(file, file.replace(clean_folder + "/", ""))

        # Upload to bucket
        blob = bucket.blob("clean.zip")
        print("uploading file", zip_file)
        blob.upload_from_filename(zip_file)
    if args.prepare:
        print("Prepare dataset for training")

        # Check if clean exists
        if not os.path.exists(clean_folder):
            # Download zip file
            source_blob_name = "clean.zip"
            destination_file_name = os.path.join(dataset_folder, "clean.zip")
            blob = bucket.blob(source_blob_name)
            blob.download_to_filename(destination_file_name)

            # Unzip data
            with zipfile.ZipFile(destination_file_name) as zfile:
                zfile.extractall(clean_folder)

        # Create TF records
        tfrecords.create_tfrecords(clean_folder, tfrecords_folder)

        # Zip dataset
        tfrecords_list = glob(tfrecords_folder + "/*")
        zip_file = os.path.join(dataset_folder, "tfrecords.zip")
        with zipfile.ZipFile(zip_file, "w") as zip:
            for file in tfrecords_list:
                zip.write(file, file.replace(dataset_folder + "/", ""))

        # Upload to bucket
        blob = bucket.blob("tfrecords.zip")
        print("uploading file", zip_file)
        blob.upload_from_filename(zip_file)
    if args.test:
        print("Test method")


if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal 'python cli.py --help', it will provide the description
    parser = argparse.ArgumentParser(description="Data Collector CLI")

    parser.add_argument(
        "-c",
        "--clean",
        action="store_true",
        help="whether or not clean images by removing duplicates",
    )
    parser.add_argument(
        "-p",
        "--prepare",
        action="store_true",
        help="Prepare data for training by creating TF Records",
    )
    parser.add_argument(
        "-b", "--bucket", type=str, default="", help="Bucket Name to save the data"
    )
    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        help="Testing...",
    )

    args = parser.parse_args()

    main(args)
