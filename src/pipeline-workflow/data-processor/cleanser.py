import os
import shutil
import difPy
import tensorflow as tf


def remove_duplicates(raw_folder, clean_folder):
    # Clear folder
    shutil.rmtree(clean_folder, ignore_errors=True, onerror=None)

    # Make a copy of the data
    shutil.copytree(
        raw_folder,
        clean_folder,
        ignore=shutil.ignore_patterns(".DS_Store"),
    )

    # Get the label folders
    label_names = os.listdir(clean_folder)
    print("Labels:", label_names)

    # Remove duplicates
    for label in label_names:
        print("Processing label:", label)
        dif = difPy.build(os.path.join(clean_folder, label))
        search = difPy.search(dif)
        print("Search Results:\n", search.result)
        print("\nLower Quality Files:\n", search.lower_quality)

        # Delete duplicates
        search.delete(silent_del=True)


def verify_images(clean_folder):
    # Get the label folders
    label_names = os.listdir(clean_folder)
    print("Labels:", label_names)

    # Verify images
    for label in label_names:
        # Images
        image_files = os.listdir(os.path.join(clean_folder, label))
        for path in image_files:
            try:
                path = os.path.join(clean_folder, label, path)
                image = tf.io.read_file(path)
                image = tf.image.decode_jpeg(image, channels=3)
            except:
                print("Deleting invalid image format:", path)
                os.remove(path)
