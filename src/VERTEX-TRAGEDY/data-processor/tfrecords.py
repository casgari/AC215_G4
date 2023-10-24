import os
import shutil
import time
import numpy as np
import dask
from sklearn.model_selection import train_test_split
import tensorflow as tf


data_details = {
    "label_folders": {
        "oyster mushrooms": 0,
        "amanita mushrooms": 1,
        "crimini mushrooms": 2,
    },
    "labels": ["oyster", "amanita", "crimini"],
    "label2index": {"oyster": 0, "amanita": 1, "crimini": 2},
    "index2label": {0: "oyster", 1: "amanita", 2: "crimini"},
    "image_width": 224,
    "image_height": 224,
    "num_channels": 3,
}


def create_tf_example(item):
    # Read image
    image = tf.io.read_file(item[1])
    image = tf.image.decode_jpeg(image, channels=data_details["num_channels"])
    image = tf.image.resize(
        image, [data_details["image_height"], data_details["image_width"]]
    )
    image = tf.cast(image, tf.uint8)

    # Label
    label = data_details["label_folders"][item[0]]

    # Build feature dict
    feature_dict = {
        "image": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[image.numpy().tobytes()])
        ),
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


@dask.delayed
def create_shard(data, i, step_size, folder, prefix):
    print(
        "Creating shard:", (i // step_size), " from records:", i, "to", (i + step_size)
    )
    path = "{}/{}_000{}.tfrecords".format(folder, prefix, i // step_size)
    print(path)

    # Write the file
    with tf.io.TFRecordWriter(path) as writer:
        # Filter the subset of data to write to tfrecord file
        for item in data[i : i + step_size]:
            tf_example = create_tf_example(item)
            writer.write(tf_example.SerializeToString())
    return 1


def create_tf_records_parallel(data, num_shards=10, prefix="", folder="data"):
    num_records = len(data)
    step_size = num_records // num_shards + 1

    delayed_create_shard = []

    for i in range(0, num_records, step_size):
        delayed_create_shard.append(create_shard(data, i, step_size, folder, prefix))

    return delayed_create_shard


def create_tfrecords(clean_folder, tfrecords_folder):
    # Clear folder
    shutil.rmtree(tfrecords_folder, ignore_errors=True, onerror=None)
    tf.io.gfile.makedirs(tfrecords_folder)

    label_names = os.listdir(clean_folder)
    print("Label Directories:", label_names)

    # Generate a list of labels and path to images
    data_list = []
    for label in label_names:
        if label == ".DS_Store":
            continue
        # Images
        image_files = os.listdir(os.path.join(clean_folder, label))
        data_list.extend(
            [(label, os.path.join(clean_folder, label, f)) for f in image_files]
        )

        print("Full size of the dataset:", len(data_list))
        print("data_list:", data_list[:5])

    # Split data
    validation_percent = 0.2
    # Split data into train / validate
    train_xy, validate_xy = train_test_split(data_list, test_size=validation_percent)
    print("train_xy count:", len(train_xy))
    print("validate_xy count:", len(validate_xy))

    # Create TF Records for train
    start_time = time.time()
    # Split data into multiple TFRecord shards between 100MB to 200MB
    num_shards = (len(train_xy) // 1000) + 1
    delayed_functions = create_tf_records_parallel(
        train_xy, num_shards=num_shards, prefix="train", folder=tfrecords_folder
    )
    total_shards_created = dask.delayed(np.sum)(delayed_functions)
    print("Total shards created:", total_shards_created.compute())
    execution_time = (time.time() - start_time) / 60.0
    print("Execution time (mins)", execution_time)

    # Create TF Records for validation
    start_time = time.time()
    # Split data into multiple TFRecord shards between 100MB to 200MB
    num_shards = (len(validate_xy) // 1000) + 1
    delayed_functions = create_tf_records_parallel(
        validate_xy, num_shards=num_shards, prefix="val", folder=tfrecords_folder
    )
    total_shards_created = dask.delayed(np.sum)(delayed_functions)
    print("Total shards created:", total_shards_created.compute())
    execution_time = (time.time() - start_time) / 60.0
    print("Execution time (mins)", execution_time)
