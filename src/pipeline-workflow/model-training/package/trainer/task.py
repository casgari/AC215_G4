import argparse
# import os
import time
# import zipfile
from collections import Counter

import tensorflow as tf
import numpy as np
import evaluate
import wandb
from transformers import DataCollatorForTokenClassification, AutoTokenizer, TFAutoModelForTokenClassification, create_optimizer
from transformers.keras_callbacks import KerasMetricCallback
from datasets import load_dataset
# from google.cloud import storage


# train_data_name = "train_data.arrow"
# valid_data_name = "valid_data.arrow"


model_name = None
batch_size = None
learning_rate = None
num_epochs = None
bucket_name = None
wandb_key = None


def train():

    #downloading data from gcp bucket
    # storage_client = storage.Client(project=gcp_project)

    # bucket = storage_client.bucket(bucket_name)

    # train_blobs = bucket.list_blobs(prefix= tokenized_data + "/train")
    # valid_blobs = bucket.list_blobs(prefix= tokenized_data + "/validation")

    # for blob in train_blobs:

    #     if blob.name.endswith(".arrow"):
    #         blob.download_to_filename(train_data_name)

    # for blob in valid_blobs:

    #     if blob.name.endswith(".arrow"):
    #         blob.download_to_filename(valid_data_name)

    # train_data = Dataset.from_file(train_data_name)
    # valid_data = Dataset.from_file(valid_data_name)


    # Load Tokenizer and Dataset
    inspec = load_dataset("midas/inspec")
    example = inspec["train"][0]

    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

    # Preprocess
    label_list = np.unique(example["doc_bio_tags"])

    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {v: k for k, v in id2label.items()}

    print('Mapping doc_bio_tag to integer:\n\n',label2id)
    print('\nMapping integer to doc_bio_tag:\n\n',id2label)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["document"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples[f"doc_bio_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label2id[label[word_idx]]) # Convert BIO to integers for classification
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # This is our tokenized dataset (a data_dict which we will convert to a TF Dataset)
    tokenized_inspec = inspec.map(tokenize_and_align_labels, batched=True)

    # Utils for training
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, return_tensors="tf")

    seqeval = evaluate.load("seqeval")

    labels = example[f"doc_bio_tags"]

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [[id2label[l] for l in label if l != -100] for label in labels]

        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        # return metrics
        all_metrics = seqeval.compute(predictions=true_predictions, references=true_labels)
        del all_metrics['_']
        print(all_metrics)
        wandb.log(all_metrics)

        return all_metrics

    #counting how many beginning keywords, middle keywords, and non-keywords there are
    count_0s = 0
    count_1s = 0
    count_2s = 0

    for listt in tokenized_inspec["train"]["labels"]:
        count_dict = Counter(listt)
        count_0s += count_dict[0]
        count_1s += count_dict[1]
        count_2s += count_dict[2]

    #getting weights for weighted cross_entropy
    max_ = max(count_0s,count_1s,count_2s)
    weights = [max_/count_0s, max_/count_1s, max_/count_2s]

    # Login to W&B
    wandb.login(key=wandb_key)

    # Train with multi-GPU
    ## Reference: https://saturncloud.io/docs/examples/python/tensorflow/qs-multi-gpu-tensorflow/
    def train_multigpu(n_epochs, base_lr, batchsize):
        num_train_steps = (len(tokenized_inspec["train"]) // batchsize) * n_epochs
        num_labels=3

        # Set up for multi-GPU training
        strategy = tf.distribute.MirroredStrategy()
        print("Number of devices: %d" % strategy.num_replicas_in_sync)

        # Initialize W&B run
        run = wandb.init(entity="ac215-ppp", project="ppp-keyword-extraction", name=f"{model_name}-trained")

        with strategy.scope():
            model = TFAutoModelForTokenClassification.from_pretrained(
                    model_name, num_labels=num_labels, id2label=id2label, label2id=label2id)

            # We define our own optimizer (and lr_schedule which we do not use)
            optimizer, lr_schedule = create_optimizer(
                                                    init_lr=2e-5,
                                                    num_train_steps=num_train_steps,
                                                    weight_decay_rate=0.01,
                                                    num_warmup_steps=0,
                                                    )

            # Compute custom loss (CrossEntropyLoss with weights)
            def loss_fn(y_true, y_pred):
                loss = tf.nn.weighted_cross_entropy_with_logits(
                    labels=tf.one_hot(y_true, depth=num_labels),
                    logits=y_pred,
                    pos_weight=tf.constant(weights)
                )
                loss = tf.reduce_mean(loss)
                return loss

            # The model is ready for training
            model.compile(loss=loss_fn, optimizer=optimizer)

        # Load in our data as TF Datasets (with data_collator applied)
        train_ds = model.prepare_tf_dataset(
                                            tokenized_inspec["train"],
                                            shuffle=True,
                                            batch_size=batchsize,
                                            collate_fn=data_collator,
                                        ).prefetch(2).cache().shuffle(1000)

        valid_ds = model.prepare_tf_dataset(
                                            tokenized_inspec["validation"],
                                            shuffle=False,
                                            batch_size=batchsize,
                                            collate_fn=data_collator,
                                        ).prefetch(2)


        # Set up callback for end of each epoch
        metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=valid_ds)
        callbacks = [metric_callback]


        # Run training
        start = time.time()

        history = model.fit(x=train_ds, validation_data=valid_ds, epochs=n_epochs, callbacks=callbacks)

        end = time.time() - start
        print("model training time", end)
        wandb.config.update({"execution_time": end})

        # Log validation loss to Weights & Biases
        wandb.define_metric("epochs")
        wandb.define_metric("validation_loss", step_metric="epochs")
        for epoch, val_loss in enumerate(history.history["val_loss"]):
            wandb.log({"epochs" : epoch, "validation_loss": val_loss})

        # Create a W&B artifact to save the model
        trained_model_artifact = wandb.Artifact("trained_model", type="model")
        # Save the model to a specified directory (adjust the path)
        directory = "model_directory"
        model.save_pretrained(directory, saved_model=True)
        # Add the saved model to the artifact
        trained_model_artifact.add_dir("model_directory")
        # Log the artifact
        run.log_artifact(trained_model_artifact)

        # Close the W&B run
        wandb.run.finish()

        return model

    # Call the training function with specified parameters
    model_params = {
        "n_epochs": num_epochs,
        "base_lr": learning_rate,
        "batchsize": batch_size
    }

    tester_plain = train_multigpu(**model_params)

    # # Upload to bucket
    # storage_client = storage.Client()
    # bucket = storage_client.bucket(bucket_name)

    # # Get the list of audio files
    # sm_s = os.listdir(saved_model)

    # for sm in sm_s:
    #     if sm.endswith('.pb'):
    #         file_path = os.path.join(saved_model, sm)

    #         destination_blob_name = file_path 
    #         blob = bucket.blob(destination_blob_name)
    #         blob.upload_from_filename(file_path)


def main(args=None):
    print("Args:", args)

    train()
    
    print("Training Job Complete")


if __name__ == "__main__":

    # Generate the inputs arguments parser
    # if you type into the terminal 'python cli.py --help', it will provide the description
    # Setup the arguments for the trainer task
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lr", 
        dest="lr", 
        default=2e-5, 
        type=float, 
        help="Learning rate."
    )
    parser.add_argument(
        "--model_name",
        dest="model_name",
        default="distilbert-base-uncased",
        type=str,
        help="Model name",
    )
    parser.add_argument(
        "--epochs", dest="epochs", default=2, type=int, help="Number of epochs."
    )
    parser.add_argument(
        "--batch_size", dest="batch_size", default=16, type=int, help="Size of a batch."
    )
    parser.add_argument(
        "--bucket_name",
        dest="bucket_name",
        default="mega-ppp-ml-workflow",
        type=str,
        help="Bucket for data and models.",
    )
    parser.add_argument(
        "--wandb_key", 
        dest="wandb_key", 
        default="", 
        type=str, 
        help="WandB API Key"
    )
    args = parser.parse_args()
    
    model_name = args.model_name
    batch_size = args.batch_size
    learning_rate = args.lr
    num_epochs = args.epochs
    bucket_name = args.bucket_name
    wandb_key = args.wandb_key

    main(args)
