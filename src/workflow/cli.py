"""
Module that contains the command line app.

Typical usage example from command line:
        python cli.py
"""

import os
import argparse
import random
import string
from kfp import dsl
from kfp import compiler
import google.cloud.aiplatform as aip


GCP_PROJECT = os.environ["GCP_PROJECT"]
GCS_BUCKET_NAME = os.environ["GCS_BUCKET_NAME"]
BUCKET_URI = f"gs://{GCS_BUCKET_NAME}"
PIPELINE_ROOT = f"{BUCKET_URI}/pipeline_root/root"
GCS_SERVICE_ACCOUNT = os.environ["GCS_SERVICE_ACCOUNT"]

DATA_CONVERSION_IMAGE = "cbsaul/ppp-workflow:preprocess_audio_file"
TRANSCRIBE_AUDIO_IMAGE = "cbsaul/ppp-workflow:transcribe-audio"
GENERATE_QUIZ_IMAGE = "cbsaul/ppp-workflow:generate-quiz"


def generate_uuid(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def main(args=None):
    print("CLI Arguments:", args)

    if args.data_conversion:
        # Define a Container Component
        @dsl.container_component
        def data_conversion():
            container_spec = dsl.ContainerSpec(
                image=DATA_CONVERSION_IMAGE,
                command=[],
                args=[
                    "cli.py",
                    "--convert"
                ],
            )
            return container_spec
        
        # Define a Pipeline
        @dsl.pipeline
        def data_conversion_pipeline():
            data_conversion()

        # Build yaml file for pipeline
        compiler.Compiler().compile(
            data_conversion_pipeline, package_path="data_conversion.yaml"
        )

        # Submit job to Vertex AI
        aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "preprocess_audio_file-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="data_conversion.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.run(service_account=GCS_SERVICE_ACCOUNT)



    if args.transcribe_audio:
        print("Transcribe Audio")

        # Define a Container Component for data processor
        @dsl.container_component
        def transcribe_audio():
            container_spec = dsl.ContainerSpec(
                image=TRANSCRIBE_AUDIO_IMAGE,
                command=[],
                args=[
                    "cli.py",
                    "--transcribe"
                ],
            )
            return container_spec

        # Define a Pipeline
        @dsl.pipeline
        def transcribe_audio_pipeline():
            transcribe_audio()

        # Build yaml file for pipeline
        compiler.Compiler().compile(
            transcribe_audio_pipeline, package_path="transcribe_audio.yaml"
        )

        # Submit job to Vertex AI
        aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "transcribe_audio-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="transcribe_audio.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.run(service_account=GCS_SERVICE_ACCOUNT)



    if args.generate_quiz:
        # Define a Container Component
        @dsl.container_component
        def generate_quiz():
            container_spec = dsl.ContainerSpec(
                image=GENERATE_QUIZ_IMAGE,
                command=[],
                args=[
                    "cli.py",
                    "--generate"
                ],
            )
            return container_spec
        
        # Define a Pipeline
        @dsl.pipeline
        def generate_quiz_pipeline():
            generate_quiz()

        # Build yaml file for pipeline
        compiler.Compiler().compile(
            generate_quiz_pipeline, package_path="generate_quiz.yaml"
        )

        # Submit job to Vertex AI
        aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "generate_quiz-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="generate_quiz.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.run(service_account=GCS_SERVICE_ACCOUNT)


    if args.pipeline:
        # Define a Container Component
        @dsl.container_component
        def data_conversion():
            container_spec = dsl.ContainerSpec(
                image=DATA_CONVERSION_IMAGE,
                command=[],
                args=[
                    "cli.py",
                    "--convert"
                ],
            )
            return container_spec

        @dsl.container_component
        def transcribe_audio():
            container_spec = dsl.ContainerSpec(
                image=TRANSCRIBE_AUDIO_IMAGE,
                command=[],
                args=[
                    "cli.py",
                    "--transcribe"
                ],
            )
            return container_spec
        
        @dsl.container_component
        def generate_quiz():
            container_spec = dsl.ContainerSpec(
                image=GENERATE_QUIZ_IMAGE,
                command=[],
                args=[
                    "cli.py",
                    "--generate"
                ],
            )
            return container_spec

        # Define a Pipeline
        @dsl.pipeline
        def ml_pipeline():
            data_conversion_task = data_conversion().set_display_name("Data Conversion")

            transcribe_audio_task = (
                transcribe_audio()
                .set_display_name("Transcribe Audio")
                .after(data_conversion_task)
            )

            generate_quiz_task = (
                generate_quiz()
                .set_display_name("Generate Quiz")
                .after(transcribe_audio_task)
            )

        # Build yaml file for pipeline
        compiler.Compiler().compile(ml_pipeline, package_path="pipeline.yaml")

        # Submit job to Vertex AI
        aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "mega-ppp-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="pipeline.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.run(service_account=GCS_SERVICE_ACCOUNT)





    if args.sample1:
        print("Sample Pipeline 1")

        # Define Component
        @dsl.component
        def square(x: float) -> float:
            return x**2

        # Define Component
        @dsl.component
        def add(x: float, y: float) -> float:
            return x + y

        # Define Component
        @dsl.component
        def square_root(x: float) -> float:
            return x**0.5

        # Define a Pipeline
        @dsl.pipeline
        def sample_pipeline(a: float = 3.0, b: float = 4.0) -> float:
            a_sq_task = square(x=a)
            b_sq_task = square(x=b)
            sum_task = add(x=a_sq_task.output, y=b_sq_task.output)
            return square_root(x=sum_task.output).output

        # Build yaml file for pipeline
        compiler.Compiler().compile(
            sample_pipeline, package_path="sample-pipeline1.yaml"
        )

        # Submit job to Vertex AI
        aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "sample-pipeline-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="sample-pipeline1.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.run(service_account=GCS_SERVICE_ACCOUNT)


if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal 'python cli.py --help', it will provide the description
    parser = argparse.ArgumentParser(description="Workflow CLI")

    parser.add_argument(
        "-c",
        "--data_conversion",
        action="store_true",
        help="Data Conversion Mp4 to Mp3",
    )

    parser.add_argument(
        "-t",
        "--transcribe_audio",
        action="store_true",
        help="Transcribe mp3 audio file",
    )

    parser.add_argument(
        "-g",
        "--generate_quiz",
        action="store_true",
        help="Generate Quiz",
    )

    parser.add_argument(
        "-p",
        "--pipeline",
        action="store_true",
        help="Run Mega PPP Pipeline",
    )

    parser.add_argument(
        "-s1",
        "--sample1",
        action="store_true",
        help="Sample Pipeline 1",
    )

    args = parser.parse_args()

    main(args)
