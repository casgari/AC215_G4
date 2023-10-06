# List of prebuilt containers for training
# https://cloud.google.com/vertex-ai/docs/training/pre-built-containers

export UUID=$(openssl rand -hex 6)
export DISPLAY_NAME="mushroom_training_multi_gpu_job_$UUID"
export MACHINE_TYPE="n1-standard-4"
export REPLICA_COUNT=1
export EXECUTOR_IMAGE_URI="us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-12.py310:latest"
export PYTHON_PACKAGE_URI=$GCS_BUCKET_URI/mushroom-app-trainer.tar.gz
export PYTHON_MODULE="trainer.task_multi_gpu"
export ACCELERATOR_TYPE="NVIDIA_TESLA_T4"
export ACCELERATOR_COUNT=2
export GCP_REGION="us-central1" # Adjust region based on you approved quotas for GPUs

export CMDARGS="--model_name=mobilenetv2,--epochs=30,--batch_size=32,--wandb_key=$WANDB_KEY"
#export CMDARGS="--model_name=mobilenetv2,--train_base,--epochs=30,--batch_size=32,--wandb_key=$WANDB_KEY"
#export CMDARGS="--model_name=tfhub_mobilenetv2,--epochs=30,--batch_size=32,--wandb_key=$WANDB_KEY"
#export CMDARGS="--model_name=tfhub_mobilenetv2,--train_base,--epochs=30,--batch_size=32,--wandb_key=$WANDB_KEY"

gcloud ai custom-jobs create \
  --region=$GCP_REGION \
  --display-name=$DISPLAY_NAME \
  --python-package-uris=$PYTHON_PACKAGE_URI \
  --worker-pool-spec=machine-type=$MACHINE_TYPE,replica-count=$REPLICA_COUNT,accelerator-type=$ACCELERATOR_TYPE,accelerator-count=$ACCELERATOR_COUNT,executor-image-uri=$EXECUTOR_IMAGE_URI,python-module=$PYTHON_MODULE \
  --args=$CMDARGS