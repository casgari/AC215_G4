from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = [
    "wandb==0.15.11",
    "tensorflow-hub==0.14.0",
    "tensorflow",
    "numpy",
    "evaluate",
    "transformers",
    "datasets",
    "google-cloud-storage",
    "python-json-logger",
    "seqeval"
]


setup(
    name="model-trainer",
    version="0.0.1",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    description="Model Trainer",
)
