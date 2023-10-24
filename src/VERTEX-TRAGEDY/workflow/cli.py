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
from model import model_training, model_deploy


GCP_PROJECT = os.environ["GCP_PROJECT"]
GCS_BUCKET_NAME = os.environ["GCS_BUCKET_NAME"]
BUCKET_URI = f"gs://{GCS_BUCKET_NAME}"
PIPELINE_ROOT = f"{BUCKET_URI}/pipeline_root/root"
GCS_SERVICE_ACCOUNT = os.environ["GCS_SERVICE_ACCOUNT"]
GCS_PACKAGE_URI = os.environ["GCS_PACKAGE_URI"]
GCP_REGION = os.environ["GCP_REGION"]

# DATA_COLLECTOR_IMAGE = "gcr.io/ac215-project/mushroom-app-data-collector"
DATA_CONVERSION_IMAGE = "cvanamburg/mega-ppp-data-conversion-v4"
AUDIO_TRANSCRIPTION_IMAGE = "cvanamburg/mega-ppp-audio-transcription"
QUIZ_GENERATION_IMAGE = "cvanamburg/mega-ppp-quiz-generation"
DATA_COLLECTOR_IMAGE = "cvanamburg/mushroom-app-data-collector"
DATA_PROCESSOR_IMAGE = "dlops/mushroom-app-data-processor"
KEYWORD_EXTRACTION_IMAGE = "the20thduck/ppp-workflow:latest"


def generate_uuid(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def main(args=None):
    print("CLI Arguments:", args)

    if args.data_conversion:
        
        @dsl.container_component
        def data_conversion():
            container_spec = dsl.ContainerSpec(
                image=DATA_CONVERSION_IMAGE,
                command=[],
                args=[
                    "cli.py",
                    "--convert"
                ]
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
        DISPLAY_NAME = "mega-ppp-data-conversion-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="data_conversion.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.run(service_account=GCS_SERVICE_ACCOUNT)
    
    if args.audio_transcription:
        
        @dsl.container_component
        def audio_transcription():
            container_spec = dsl.ContainerSpec(
                image=AUDIO_TRANSCRIPTION_IMAGE,
                command=[],
                args=[
                    "cli.py",
                    "--transcribe"
                ]
            )
            return container_spec

        # Define a Pipeline
        @dsl.pipeline
        def audio_transcription_pipeline():
            audio_transcription()

        # Build yaml file for pipeline
        compiler.Compiler().compile(
            audio_transcription_pipeline, package_path="audio_transcription.yaml"
        )

        # Submit job to Vertex AI
        aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "mega-ppp-audio_transcription-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="audio_transcription.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.run(service_account=GCS_SERVICE_ACCOUNT)


    if args.keyword_extraction:
        
        @dsl.container_component
        def keyword_extraction():
            container_spec = dsl.ContainerSpec(
                image=KEYWORD_EXTRACTION_IMAGE,
                command=[],
                args=[
                    "cli.py",
                    "-p"
                ]
            )
            return container_spec

        # Define a Pipeline
        @dsl.pipeline
        def keyword_extraction_pipeline():
            keyword_extraction()

        # Build yaml file for pipeline
        compiler.Compiler().compile(
            keyword_extraction_pipeline, package_path="keyword_extraction.yaml"
        )

        # Submit job to Vertex AI
        aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "mega-ppp-keyword-extraction-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="keyword_extraction.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.run(service_account=GCS_SERVICE_ACCOUNT)

    if args.quiz_generation:
        
        @dsl.container_component
        def quiz_generation():
            container_spec = dsl.ContainerSpec(
                image=QUIZ_GENERATION_IMAGE,
                command=[],
                args=[
                    "cli.py",
                    "--generate"
                ]
            )
            return container_spec

        # Define a Pipeline
        @dsl.pipeline
        def quiz_generation_pipeline():
            quiz_generation()

        # Build yaml file for pipeline
        compiler.Compiler().compile(
            quiz_generation_pipeline, package_path="quiz_generation.yaml"
        )

        # Submit job to Vertex AI
        aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "mega-ppp-quiz_generation-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="quiz_generation.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.run(service_account=GCS_SERVICE_ACCOUNT)







    if args.data_collector:
        # Define a Container Component
        @dsl.container_component
        def data_collector():
            container_spec = dsl.ContainerSpec(
                image=DATA_COLLECTOR_IMAGE,
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

        # Define a Pipeline
        @dsl.pipeline
        def data_collector_pipeline():
            data_collector()

        # Build yaml file for pipeline
        compiler.Compiler().compile(
            data_collector_pipeline, package_path="data_collector.yaml"
        )

        # Submit job to Vertex AI
        aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "mushroom-app-data-collector-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="data_collector.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.run(service_account=GCS_SERVICE_ACCOUNT)

    if args.data_processor:
        print("Data Processor")

        # Define a Container Component for data processor
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
        def data_processor_pipeline():
            data_processor()

        # Build yaml file for pipeline
        compiler.Compiler().compile(
            data_processor_pipeline, package_path="data_processor.yaml"
        )

        # Submit job to Vertex AI
        aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "mushroom-app-data-processor-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="data_processor.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.run(service_account=GCS_SERVICE_ACCOUNT)

    if args.model_training:
        print("Model Training")

        # Define a Pipeline
        @dsl.pipeline
        def model_training_pipeline():
            model_training(
                project=GCP_PROJECT,
                location=GCP_REGION,
                staging_bucket=GCS_PACKAGE_URI,
                bucket_name=GCS_BUCKET_NAME,
            )

        # Build yaml file for pipeline
        compiler.Compiler().compile(
            model_training_pipeline, package_path="model_training.yaml"
        )

        # Submit job to Vertex AI
        aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "mushroom-app-model-training-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="model_training.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.run(service_account=GCS_SERVICE_ACCOUNT)

    if args.model_deploy:
        print("Model Deploy")

        # Define a Pipeline
        @dsl.pipeline
        def model_deploy_pipeline():
            model_deploy(
                bucket_name=GCS_BUCKET_NAME,
            )

        # Build yaml file for pipeline
        compiler.Compiler().compile(
            model_deploy_pipeline, package_path="model_deploy.yaml"
        )

        # Submit job to Vertex AI
        aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "mushroom-app-model-deploy-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="model_deploy.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.run(service_account=GCS_SERVICE_ACCOUNT)

    if args.pipeline:
        @dsl.container_component
        def data_conversion():
            container_spec = dsl.ContainerSpec(
                image=DATA_CONVERSION_IMAGE,
                command=[],
                args=[
                    "cli.py",
                    "--convert"
                ]
            )
            return container_spec
        
        @dsl.container_component
        def audio_transcription():
            container_spec = dsl.ContainerSpec(
                image=AUDIO_TRANSCRIPTION_IMAGE,
                command=[],
                args=[
                    "cli.py",
                    "--transcribe"
                ]
            )
            return container_spec
        
        @dsl.container_component
        def keyword_extraction():
            container_spec = dsl.ContainerSpec(
                image=KEYWORD_EXTRACTION_IMAGE_IMAGE,
                command=[],
                args=[
                    "cli.py",
                    "-p"
                ]
            )
            return container_spec

        @dsl.container_component
        def quiz_generation():
            container_spec = dsl.ContainerSpec(
                image=QUIZ_GENERATION_IMAGE,
                command=[],
                args=[
                    "cli.py",
                    "--generate"
                ]
            )
            return container_spec






        # Define a Container Component for data collector
        @dsl.container_component
        def data_collector():
            container_spec = dsl.ContainerSpec(
                image=DATA_COLLECTOR_IMAGE,
                command=[],
                args=[
                    "cli.py",
                    "--search",
                    "--nums 50",
                    "--query oyster+mushrooms crimini+mushrooms amanita+mushrooms",
                    f"--bucket {GCS_BUCKET_NAME}",
                ],
            )
            return container_spec

        # Define a Container Component for data processor
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
            # Data Conversion
            data_conversion_task = data_conversion().set_display_name("Data Conversion")
            # Data Collector
            audio_transcription_task = (
                audio_transcription()
                .set_display_name("Audio Transcription")
                .after(data_conversion_task)
            )
            # Data Processor
            quiz_generation_task = (
                quiz_generation()
                .set_display_name("Quiz Generation")
                .after(audio_transcription_task)
            )
            keyword_extraction_task = (
                keyword_extraction()
                .set_display_name("Keyword Extraction")
                .after(quiz_generation_task)
            )
            # # Model Training
            # model_training_task = (
            #     model_training(
            #         project=GCP_PROJECT,
            #         location=GCP_REGION,
            #         staging_bucket=GCS_PACKAGE_URI,
            #         bucket_name=GCS_BUCKET_NAME,
            #         epochs=15,
            #         batch_size=16,
            #         model_name="mobilenetv2",
            #         train_base=False,
            #     )
            #     .set_display_name("Model Training")
            #     .after(data_processor_task)
            # )
            # # Model Deployment
            # model_deploy_task = (
            #     model_deploy(
            #         bucket_name=GCS_BUCKET_NAME,
            #     )
            #     .set_display_name("Model Deploy")
            #     .after(model_training_task)
            # )

        # Build yaml file for pipeline
        compiler.Compiler().compile(ml_pipeline, package_path="pipeline.yaml")

        # Submit job to Vertex AI
        aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "mushroom-app-pipeline-" + job_id
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
        "-e",
        "--keyword_extraction",
        action="store_true",
        help="Run just the Keyword Extraction",
    )
    parser.add_argument(
        "-k",
        "--data_conversion",
        action="store_true",
        help="Run just the Data Conversion",
    )
    parser.add_argument(
        "-r",
        "--audio_transcription",
        action="store_true",
        help="Run just the Audio Transcription",
    )
    parser.add_argument(
        "-g",
        "--quiz_generation",
        action="store_true",
        help="Run just the Quiz Generation",
    )
    parser.add_argument(
        "-c",
        "--data_collector",
        action="store_true",
        help="Run just the Data Collector",
    )
    parser.add_argument(
        "-p",
        "--data_processor",
        action="store_true",
        help="Run just the Data Processor",
    )
    parser.add_argument(
        "-t",
        "--model_training",
        action="store_true",
        help="Run just Model Training",
    )
    parser.add_argument(
        "-d",
        "--model_deploy",
        action="store_true",
        help="Run just Model Deployment",
    )
    parser.add_argument(
        "-w",
        "--pipeline",
        action="store_true",
        help="Mushroom App Pipeline",
    )
    parser.add_argument(
        "-s1",
        "--sample1",
        action="store_true",
        help="Sample Pipeline 1",
    )

    args = parser.parse_args()

    main(args)
