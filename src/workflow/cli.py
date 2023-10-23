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
        DISPLAY_NAME = "mega-ppp-data-conversion"

        job_id = generate_uuid()
        DISPLAY_NAME = "mega-ppp-data-conversion-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="data_conversion.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.run(service_account=GCS_SERVICE_ACCOUNT)

    if args.pipeline:
        # Define a Container Component
        @dsl.container_component
        def data_collector():
            container_spec = dsl.ContainerSpec(
                image=DATA_CONVERSION_IMAGE,
                command=[],
                args=[
                    "cli.py",
                    "--search",
                    "--nums 10",
                    "--query oyster+mushrooms crimini+mushrooms amanita+mushrooms",
                    f"--bucket {GCS_BUCKET_NAME}",
                ],
            )
            return container_spec

        @dsl.container_component
        def data_processor():
            container_spec = dsl.ContainerSpec(
                image=DATA_PROCESSOR_IMAGE,
                command=[],
                args=[
                    "cli.py",
                    "--clean",
                    "--prepare",
                    f"--bucket {GCS_BUCKET_NAME}",
                ],
            )
            return container_spec

        # Define a Pipeline
        @dsl.pipeline
        def ml_pipeline():
            data_collector_task = data_collector().set_display_name("Data Collector")

            data_processor_task = (
                data_processor()
                .set_display_name("Data Processor")
                .after(data_collector_task)
            )

        # Build yaml file for pipeline
        compiler.Compiler().compile(ml_pipeline, package_path="pipeline.yaml")

        # Submit job to Vertex AI
        aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)
        DISPLAY_NAME = "mushroom-app-pipeline"

        job_id = generate_uuid()
        DISPLAY_NAME = "mushroom-app-pipeline-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="pipeline.yaml",
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
        "-w",
        "--pipeline",
        action="store_true",
        help="Mushroom App Pipeline",
    )

    args = parser.parse_args()

    main(args)
