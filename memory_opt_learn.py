# Imports for preprocessing
from sklearn.metrics import classification_report
import tensorflow as tf
import numpy as np
import pathlib
import os
import pandas as pd

# Imports for deep-learning
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, ReLU
from keras.optimizers import Adam

AUTOTUNE = tf.data.AUTOTUNE

base_dir = "/home/fhase/Desktop/Dataset_S"


def get_class_names():
    data_dir = pathlib.Path(base_dir).with_suffix("")
    class_names = np.array(sorted([item.name for item in data_dir.glob(
        '*') if item.name != "NeuralNetwork_Results.xlsx"]))
    return class_names


def get_data(batch_size, img_height, img_width, sample_size):
    data_dir = pathlib.Path(base_dir).with_suffix("")

    list_ds = tf.data.Dataset.list_files(
        str(data_dir/'*/database/images/*'), shuffle=False)

    image_count = len(list(data_dir.glob('*/database/images/*.jpg')))

    list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

    class_names = np.array(sorted([item.name for item in data_dir.glob(
        '*') if item.name != "NeuralNetwork_Results.xlsx"]))

    list_ds = list_ds.take(sample_size)
    image_count = sample_size

    val_size = int(image_count * 0.2)

    train_ds = list_ds.skip(val_size)
    val_ds = list_ds.take(val_size)

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
        return tf.image.resize(img, [img_height, img_width])

    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    def configure_for_performance(ds):
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    train_ds = configure_for_performance(train_ds)
    val_ds = configure_for_performance(val_ds)
    return train_ds, val_ds



