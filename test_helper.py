import pathlib
import numpy as np
import tensorflow as tf
import os

AUTOTUNE = tf.data.AUTOTUNE

base_dir = "/home/fhase/Desktop/Dataset_S"


def get_test_data(image_size, test_size) -> tf.data.Dataset:
    data_dir = pathlib.Path(base_dir).with_suffix("")

    class_names = np.array(sorted([item.name for item in data_dir.glob(
        '*') if item.name != "NeuralNetwork_Results.xlsx"]))

    list_ds = []

    for c in class_names:
        list_ds.append(tf.data.Dataset.list_files(
            str(data_dir/'{c}/query/images/*'.format(c=c)), shuffle=True))

    biglist = []
    for d in list_ds:
        d = d.take(test_size)
        for t in d.as_numpy_iterator():
            biglist.append(t)

    list_ds = tf.data.Dataset.from_tensor_slices(biglist)
    for t in list_ds:
        print(t)

    def get_label(file_path):
        # Convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        one_hot = parts[-4] == class_names
        # Integer encode the label
        return tf.argmax(one_hot)

    def process_path(file_path):
        label = get_label(file_path)
        # Load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img, label

    def decode_img(img):
        # Convert the compressed string to a 3D uint8 tensor
        img = tf.io.decode_jpeg(img, channels=3)
        # Resize the image to the desired size
        return tf.image.resize(img, [image_size, image_size])

    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    test_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    return list(test_ds)

if "__name__" == "__main__":
    get_test_data(100, 10)