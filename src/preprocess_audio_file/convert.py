"""
Module that contains the command line app.
"""
import os
import argparse
import shutil
from google.cloud import storage
from moviepy.editor import VideoFileClip

# Generate the inputs arguments parser
parser = argparse.ArgumentParser(description="Command description.")

gcp_project = "AC215 Group 4"
bucket_name = "mega-ppp"
input_videos = "input_videos"
audio_files = "audio_files"



def makedirs():
    os.makedirs(input_videos, exist_ok=True)
    os.makedirs(audio_files, exist_ok=True)


def download():
    print("download")

    # Clear
    shutil.rmtree(input_videos, ignore_errors=True, onerror=None)
    makedirs()

    storage_client = storage.Client(project=gcp_project)

    bucket = storage_client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=input_videos + "/")
    for blob in blobs:
        print(blob.name)
        if blob.name.endswith(".mp4"):
            blob.download_to_filename(blob.name)


def convert():
    print("convert")
    makedirs()

    # Get the list of video files
    video_files = os.listdir(input_videos)

    for video_path in video_files:
        uuid = video_path.replace(".mp4", "")
        video_path = os.path.join(input_videos, video_path)
        converted_file = os.path.join(audio_files, uuid + ".mp3")

        if os.path.exists(converted_file):
            continue

        video = VideoFileClip(video_path)
        video.audio.write_audiofile(converted_file)


def upload():
    print("upload")
    makedirs()

    # Upload to bucket
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Get the list of audio files
    converted_audio_files = os.listdir(audio_files)

    for audio_file in converted_audio_files:
        file_path = os.path.join(audio_files, audio_file)

        destination_blob_name = file_path
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(file_path)


def main(args=None):
    print("Args:", args)

    if args.download:
        download()
    if args.convert:
        convert()
    if args.upload:
        upload()


if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal 'python cli.py --help', it will provide the description
    parser = argparse.ArgumentParser(description="Convert video to audio")

    parser.add_argument(
        "-d",
        "--download",
        action="store_true",
        help="Download input video from GCS bucket",
    )

    parser.add_argument("-c", "--convert", action="store_true", help="Convert mp4 to mp3")

    parser.add_argument(
        "-u",
        "--upload",
        action="store_true",
        help="Upload converted audio file to GCS bucket",
    )

    args = parser.parse_args()

    main(args)
