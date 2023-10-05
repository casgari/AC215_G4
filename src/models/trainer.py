"""
Module that trains the model
"""
import argparse
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import evaluate
import torch
from collections import Counter
from datasets import Dataset
from gcsfs import GCSFileSystem
from google.cloud import storage
import wandb
import time


# Generate the inputs arguments parser
parser = argparse.ArgumentParser(description="Command description.")

gcp_project = "AC215 Group 4"
bucket_name = "mega-ppp"
tokenized_data = "tokenized_data"
saved_model = "saved_model"

train_data_name = "train_data.arrow"
valid_data_name = "valid_data.arrow"


model_checkpoint = None
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


    if torch.cuda.is_available():
        device = torch.device("cuda")

    else:
        device = torch.device("cpu")


    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    metric = evaluate.load("seqeval")

    def compute_metrics(preds):
        logits, labels = preds
        predictions = np.argmax(logits, axis=-1)

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [[id2label[l] for l in label if l != -100] for label in labels]

        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
        return all_metrics


    wandb.login(key=wandb_key)

    model_name = model_checkpoint.split("/")[-1]


    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint,
                                                            id2label=id2label,
                                                            label2id=label2id)
    model = model.to(device)

    args = TrainingArguments(
    f"{model_name}_finetuned_keyword_extract",
    evaluation_strategy = "epoch",
    logging_strategy = 'epoch',
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs= num_epochs,
    lr_scheduler_type='linear',
    weight_decay=0.01,
    seed=0
    )
    
    #counting how many beginning keywords, middle keywords, and non-keywords there are
    count_0s = 0
    count_1s = 0
    count_2s = 0

    for listt in train_data["labels"]:
        count_dict = Counter(listt)
        count_0s += count_dict[0]
        count_1s += count_dict[1]
        count_2s += count_dict[2]

    #getting weights for weighted cross_entropy
    max_ = max(count_0s,count_1s,count_2s)
    weights = [max_/count_0s, max_/count_1s, max_/count_2s]

    #defining loss function
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels").to(model.device)
            # forward pass
            outputs = model(**inputs)
            logits = outputs.get("logits").to(model.device)
            # compute custom loss (suppose one has 3 labels with different weights)
            loss_fct = torch.nn.CrossEntropyLoss(weight= torch.tensor(weights).to(device))
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss
    

    # Initialize a W&B run
    wandb.init(
        project = 'ppp-keyword-extraction',
        config = {
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "model_name": model_name
        },
        name = model_name
    )


    # Train model
    start_time = time.time()


    trainer = CustomTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer)
    trainer.train()

    execution_time = (time.time() - start_time)/60.0
    print("Training execution time (mins)",execution_time)

    # Update W&B
    wandb.config.update({"execution_time": execution_time})
    # Close the W&B run
    wandb.run.finish()


    fs = GCSFileSystem(project = gcp_project)
    with fs.open("gs://" + bucket_name + "/" + saved_model + '/' + model_name, 'wb') as f:
        torch.save(model, f)


def main(args=None):
    print("Args:", args)

    if args.train:
        train()



if __name__ == "__main__":

    # Generate the inputs arguments parser
    # if you type into the terminal 'python cli.py --help', it will provide the description
    parser = argparse.ArgumentParser(description="tokenize inspec dataset")
    

    parser.add_argument("-t", "--train", action="store_true", help="train model")
    parser.add_argument("-m", "--model_checkpoint", type = str, default = "distilbert-base-uncased", help="model checkpoint name")
    parser.add_argument("-b", "--batch_size", type = int, default = 8, help="model checkpoint name")
    parser.add_argument("-lr", "--learning_rate", type = float, default = 4e-6, help="model checkpoint name")
    parser.add_argument("-ne", "--num_epochs", type = int, default = 6, help="model checkpoint name")
    parser.add_argument("-wandb_key", dest="wandb_key", default="16", type=str, help="WandB API Key")
    args = parser.parse_args()
    
    model_checkpoint = args.model_checkpoint
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    wandb_key = args.wandb_key



    main(args)
