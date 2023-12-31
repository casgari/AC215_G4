"""
Module that contains the command line app.
"""
import os
import argparse
import shutil
from google.cloud import storage
from moviepy.editor import VideoFileClip
import dask
from dask.distributed import Client

# Generate the inputs arguments parser
parser = argparse.ArgumentParser(description="Command description.")



input_videos = "input_videos"
audio_files = "audio_files"

GCP_PROJECT = os.getenv("GCP_PROJECT")
# GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GCS_BUCKET_NAME = "mega-ppp-ml-workflow"


def makedirs():
    os.makedirs(input_videos, exist_ok=True)
    os.makedirs(audio_files, exist_ok=True)


def download():
    print("download")

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(GCS_BUCKET_NAME)
    # bucket = storage_client.get_bucket(bucket_n) 

    # Clear
    shutil.rmtree(input_videos, ignore_errors=True, onerror=None)
    makedirs()


    blobs = bucket.list_blobs(prefix=input_videos + "/")
    for blob in blobs:
        print(blob.name)
        if blob.name.endswith(".mp4"):
            blob.download_to_filename(blob.name)

@dask.delayed
def dask_convert(video_path, audio_files):
    uuid = video_path.replace(".mp4", "")
    video_path = os.path.join(input_videos, video_path)
    converted_file = os.path.join(audio_files, uuid + ".mp3")

    if os.path.exists(converted_file):
        return

    video = VideoFileClip(video_path)
    video.audio.write_audiofile(converted_file)
    return converted_file

@dask.delayed
def dask_upload(audio_file, bucket_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    file_path = os.path.join(audio_files, audio_file)
    destination_blob_name = file_path
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(file_path)
    return destination_blob_name

def convert():
    print("convert")
    makedirs()

    # Get the list of video files
    video_files = os.listdir(input_videos)
    results = [dask_convert(video_path, audio_files) for video_path in video_files]

    # Using dask compute to start the computation
    dask.compute(*results)

def upload():
    print("upload")
    makedirs()

    # Get the list of audio files
    converted_audio_files = os.listdir(audio_files)

    results = [dask_upload(file_path, GCS_BUCKET_NAME) for file_path in converted_audio_files]

    # Using dask compute to start the computation
    dask.compute(*results)


def main(args=None):
    print("Args:", args)
    if args.convert:
        download()
        client = Client()  # starts local cluster, uses all available cores
        convert()
        client.close()
        client = Client()
        upload()
        client.close()



if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal 'python cli.py --help', it will provide the description
    parser = argparse.ArgumentParser(description="Convert video to audio")

    parser.add_argument("-c", "--convert", action="store_true", help="Convert mp4 to mp3")

    args = parser.parse_args()

    main(args)
