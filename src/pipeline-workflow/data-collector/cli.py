"""
Module that contains the command line app.

Typical usage example from command line:
        python cli.py --search --nums 10 --query "oyster mushrooms" "crimini mushrooms" "amanita mushrooms"
"""

import os
import argparse
from glob import glob
import zipfile
from google.cloud import storage
import downloader

dataset_folder = os.path.join("/persistent", "dataset")


def main(args=None):
    print("CLI Arguments:", args)

    # GCP_PROJECT = os.environ["GCP_PROJECT"]
    GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

    # If bucket was passed as argument
    if args.bucket != "":
        GCS_BUCKET_NAME = args.bucket
    print("GCS_BUCKET_NAME:", GCS_BUCKET_NAME)

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(GCS_BUCKET_NAME)

    # Make dirs
    os.makedirs(dataset_folder, exist_ok=True)

    if args.search:
        search_term_list = args.query

        raw_folder = os.path.join(dataset_folder, "raw")

        # Download images for each search term
        for search_term in search_term_list:
            search_term = search_term.replace("+", " ")
            print("Searching for:", search_term)
            downloader.download_bing_images(
                search_term,
                limit=args.nums,
                output_dir=raw_folder,
                adult_filter_off=False,
                force_replace=False,
                timeout=60,
                verbose=True,
            )

        # Zip dataset
        data_list = glob(raw_folder + "/*/*")
        zip_file = os.path.join(dataset_folder, "raw.zip")
        if os.path.exists(zip_file):
            os.remove(zip_file)
        with zipfile.ZipFile(zip_file, "w") as zip:
            for file in data_list:
                zip.write(file, file.replace(raw_folder + "/", ""))

        # Upload to bucket
        blob = bucket.blob("raw.zip")
        print("uploading file", zip_file)
        blob.upload_from_filename(zip_file)


if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal 'python cli.py --help', it will provide the description
    parser = argparse.ArgumentParser(description="Data Collector CLI")

    parser.add_argument(
        "-s",
        "--search",
        action="store_true",
        help="whether or not to download images",
    )
    parser.add_argument(
        "-n", "--nums", type=int, default=1, help="number of images to download"
    )
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        nargs="+",
        default="oyster mushrooms",
        help="the search query term(s) since there can be multiple",
    )
    parser.add_argument(
        "-b", "--bucket", type=str, default="", help="Bucket Name to save the data"
    )

    args = parser.parse_args()

    main(args)
