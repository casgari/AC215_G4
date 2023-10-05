"""
Module that tokenizes dataset
"""
import argparse
import numpy as np
from google.cloud import storage
from transformers import AutoTokenizer
from datasets import load_dataset
from gcsfs import GCSFileSystem


# Generate the inputs arguments parser
parser = argparse.ArgumentParser(description="Command description.")

gcp_project = "AC215 Group 4"
bucket_name = "mega-ppp"
tokenized_data = "tokenized_data"


model_checkpoint = None
label2id = None

def tokenize_words_with_corresponding_labels(sample):

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

    #truncation=True to specify to truncate sequences at the maximum length
    #is_split_into_words = True to specify that our input is already pre-tokenized (e.g., split into words)
    tokenized_inputs = tokenizer(sample["document"], truncation=True, is_split_into_words=True)

    #initialize list to store lists of labels for each sample
    labels = []

    for i, label in enumerate(sample["doc_bio_tags"]):

        #map tokens to their respective word
        #word_ids() method gets index of the word that each token comes from
        word_ids = tokenized_inputs.word_ids(batch_index=i)

        #initialize list of labels for each token in a given sample
        label_ids = []

        for word_idx in word_ids:

            #set the special tokens, [CLS] and [SEP], to -100.
            # we use -100 because it's an index that is ignored in the loss function we will use (cross entropy).
            if word_idx is None:
                label_ids.append(-100)

            #set labels for tokens
            else:
                label_ids.append(label2id[label[word_idx]])

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels

    return tokenized_inputs



def tokenize():
    print("tokenizer")
    global label2id

    dataset = load_dataset("midas/inspec", "extraction")
    train_sample = dataset["train"][0]
    label_list = np.unique(train_sample["doc_bio_tags"])
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {v: k for k, v in id2label.items()}

    tokenized_dataset = dataset.map(tokenize_words_with_corresponding_labels, batched=True)
    fs = GCSFileSystem()
    tokenized_dataset.save_to_disk("gs://" + bucket_name + "/" + tokenized_data, fs=fs)



def main(args=None):
    print("Args:", args)

    if args.tokenize:
        tokenize()



if __name__ == "__main__":

    # Generate the inputs arguments parser
    # if you type into the terminal 'python cli.py --help', it will provide the description
    parser = argparse.ArgumentParser(description="tokenize inspec dataset")
    

    parser.add_argument("-t", "--tokenize", action="store_true", help="tokenize dataset")
    parser.add_argument("-m", "--model_checkpoint", type = str, default = "distilbert-base-uncased", help="model checkpoint name")

    args = parser.parse_args()
    
    model_checkpoint = args.model_checkpoint

    main(args)
