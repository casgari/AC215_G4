"""
Module that trains the model
"""
import os
import argparse
import numpy as np
import wandb
import evaluate
import tensorflow as tf
import time
from transformers import AutoTokenizer
from datasets import Dataset
from transformers import DataCollatorForTokenClassification
from transformers import TFAutoModelForTokenClassification
from transformers.keras_callbacks import KerasMetricCallback
from transformers import create_optimizer
from google.cloud import storage



# Generate the inputs arguments parser
parser = argparse.ArgumentParser(description="Command description.")

gcp_project = "AC215 Group 4"
bucket_name = "mega-ppp"
tokenized_data = "tokenized_data"
saved_model = "saved_model"

train_data_name = "train_data.arrow"
valid_data_name = "valid_data.arrow"


model_name = None
batch_size = None
learning_rate = None
num_epochs = None
wandb_key = None


def train():

    #downloading data from gcp bucket
    storage_client = storage.Client(project=gcp_project)

    bucket = storage_client.bucket(bucket_name)

    train_blobs = bucket.list_blobs(prefix= tokenized_data + "/train")
    valid_blobs = bucket.list_blobs(prefix= tokenized_data + "/validation")

    for blob in train_blobs:

        if blob.name.endswith(".arrow"):
            blob.download_to_filename(train_data_name)

    for blob in valid_blobs:

        if blob.name.endswith(".arrow"):
            blob.download_to_filename(valid_data_name)

    train_data = Dataset.from_file(train_data_name)
    valid_data = Dataset.from_file(valid_data_name)

    
    
    train_sample = train_data[0]
    label_list = np.unique(train_sample["doc_bio_tags"])

    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {v: k for k, v in id2label.items()}



    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, return_tensors="tf")

    seqeval = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        log_results = {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
        wandb.log(log_results)

        return log_results


    wandb.login(key=wandb_key)


    def train_multigpu(n_epochs, base_lr, batchsize):
        num_train_steps = (len(train_data) // batchsize) * n_epochs

        # Set up for multi-GPU training
        strategy = tf.distribute.MirroredStrategy()
        print("Number of devices: %d" % strategy.num_replicas_in_sync)

        with strategy.scope():
            model = TFAutoModelForTokenClassification.from_pretrained(
                    model_name, num_labels=len(label_list), id2label=id2label, label2id=label2id)

            # We define our own optimizer (and lr_schedule which we do not use)
            optimizer, lr_schedule = create_optimizer(
                                                    init_lr=learning_rate,
                                                    num_train_steps=num_train_steps,
                                                    weight_decay_rate=0.01,
                                                    num_warmup_steps=0,
                                                    )
            # The model is ready for training
            model.compile(optimizer=optimizer, metrics=["accuracy"])

        # Load in our data as TF Datasets (with data_collator applied)
        train_ds = model.prepare_tf_dataset(
                                            train_data,
                                            shuffle=True,
                                            batch_size=batchsize,
                                            collate_fn=data_collator,
                                        ).prefetch(2).cache().shuffle(1000)

        valid_ds = model.prepare_tf_dataset(
                                            valid_data,
                                            shuffle=False,
                                            batch_size=batchsize,
                                            collate_fn=data_collator,
                                        ).prefetch(2)

        # Set up a callback for metrics at the end of every epoch
        metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=valid_ds)
        callbacks = [metric_callback]

        # Run training
        start = time.time()

        model.fit(x=train_ds, validation_data=valid_ds, epochs=n_epochs, callbacks=callbacks)

        end = time.time() - start
        print("model training time", end)

        # Save the trained model
        path = saved_model + "/" 
        tf.keras.models.save_model(model, path)
        print(f"Saved model to {path}")
    

    # Initialize a W&B run
    wandb.init(
        project = 'ppp-keyword-extraction',
        config = {
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "model_name": model_name,
        },
        name = model_name
    )

    # Call the training function with specified parameters
    model_params = {
        "n_epochs": num_epochs,
        "base_lr": learning_rate,
        "batchsize": batch_size
    }

    tester_plain = train_multigpu(**model_params)

    # Close the W&B run
    wandb.run.finish()

    print('Uploading saved model to GCP')

    # Upload to bucket
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Get the list of audio files
    sm_s = os.listdir(saved_model)

    for sm in sm_s:
        file_path = os.path.join(saved_model, sm)

        destination_blob_name = file_path 
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(file_path)


def main(args=None):
    print("Args:", args)

    if args.train:
        train()



if __name__ == "__main__":

    # Generate the inputs arguments parser
    # if you type into the terminal 'python cli.py --help', it will provide the description
    parser = argparse.ArgumentParser(description="tokenize inspec dataset")
    

    parser.add_argument("-t", "--train", action="store_true", help="train model")
    parser.add_argument("-m", "--model_name", type = str, default = "distilbert-base-uncased", help="model checkpoint name")
    parser.add_argument("-b", "--batch_size", type = int, default = 16, help="batch size")
    parser.add_argument("-lr", "--learning_rate", type = float, default = 2e-5, help="learning rate")
    parser.add_argument("-ne", "--num_epochs", type = int, default = 6, help="number of epochs")
    parser.add_argument("-wandb_key", dest="wandb_key", default="16", type=str, help="WandB API Key")
    args = parser.parse_args()
    
    model_name = args.model_name
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    wandb_key = args.wandb_key



    main(args)
