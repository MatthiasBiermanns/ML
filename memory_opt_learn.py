# Imports for preprocessing
import logging
import traceback
from sklearn.metrics import classification_report
import tensorflow as tf
import numpy as np
import pathlib
import os
import pandas as pd

# Imports for deep-learning
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.optimizers import Adam

from test_helper import get_test_data

AUTOTUNE = tf.data.AUTOTUNE

base_dir = "/home/fhase/Desktop/Dataset_S"
batch_size = 1


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


def generate_model(layers):
    model = Sequential()
    for layer in layers:
        model.add(layer)
    return model


def train_model(model, learning_rate, train_ds, val_ds, epoch_range):
    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(
    ), metrics=['accuracy'])

    history = model.fit(train_ds, epochs=epoch_range, validation_data=val_ds)

    return model, history


def get_metrics(model, labels, x_val, y_val):
    predictionOdds = model.predict(x_val)
    predictions = []
    for pred in predictionOdds:
        predictions.append(list(pred).index(max(pred)))

    predictions = np.array(predictions)
    predictions = predictions.reshape(1, -1)[0]
    return classification_report(y_val, predictions, target_names=labels)


def run_neural_network(img_size, processed_images, labels, epoch_range, learning_rate, layers):

    batch_size = pow(2, processed_images // (32 + processed_images // 10))
    train_ds, val_ds = get_data(
        batch_size, img_height=img_size, img_width=img_size, sample_size=processed_images)

    model = generate_model(layers)
    print(model.summary())

    model, history = train_model(
        model, learning_rate, train_ds, val_ds, epoch_range)

    test_ds = get_test_data(image_size=img_size,
                            test_size=10)
    x_val = list(map(lambda x: x[0].numpy().tolist(), test_ds))
    y_val = list(map(lambda x: x[1].numpy(), test_ds))
    metrics = get_metrics(model, labels, x_val, y_val)
    print(metrics)

    return history, metrics


def main():
    resultPath = os.path.join(base_dir, 'NeuralNetwork_Results.xlsx')
    cities = get_class_names()
    image_count = [1000]  # , 1000, 2000]
    resolution = [225]  # , 450, 1000]
    learning_rate = [0.000001]  # , 0.00001, 0.0001]
    epoch_range = 100

    results = pd.DataFrame(columns=['ID', 'Netzwerk', 'Anzahl Bilder', 'Bildauflösung (px)', 'Lernquote',
                                    'Train Acc', 'Val Acc', 'Train Loss', 'Val Loss', 'Metriken'])
    results = results.set_index('ID')
    rowId = 0
    try:
        for images in image_count:
            for res in resolution:
                if images > 1000 and res > 450:
                    continue
                for lr in learning_rate:
                    filters = res // 7
                    # needs res for initialization
                    network1 = [
                        Conv2D(filters, 3, padding="same", activation="tanh",
                               input_shape=(res, res, 3)),
                        Conv2D(filters, 3, padding="same", activation="tanh"),
                        MaxPool2D(),
                        Conv2D(filters*2, 3, padding="same",
                               activation="tanh"),
                        MaxPool2D(),
                        Dropout(0.4),
                        Flatten(),
                        Dense(128, activation="tanh"),
                        Dense(len(cities), activation="softmax")
                    ]
                    network2 = [
                        Conv2D(32, 3, padding="same", activation="relu",
                               input_shape=(res, res, 3)),
                        MaxPool2D(),
                        Conv2D(32, 3, padding="same", activation="relu"),
                        MaxPool2D(),
                        Conv2D(64, 3, padding="same", activation="relu"),
                        MaxPool2D(),
                        Dropout(0.4),
                        Flatten(),
                        Dense(128, activation="relu"),
                        Dense(len(cities), activation="softmax")
                    ]
                    network3 = [
                        Conv2D(32, (3, 3), activation='relu', input_shape=(res, res, 3)),
                        MaxPool2D((2, 2)),
                        Conv2D(64, (3, 3), activation='relu'),
                        MaxPool2D((2, 2)),
                        Conv2D(128, (3, 3), activation='relu'),
                        MaxPool2D((2, 2)),
                        Conv2D(256, (3, 3), activation='relu'),
                        MaxPool2D((2, 2)),
                        Flatten(),
                        Dense(256, activation='relu'),
                        Dropout(0.5),
                        Dense(128, activation='relu'),
                        Dropout(0.5),
                        Dense(len(cities), activation='softmax')
                    ]
                    networks = [network3]  # , network2]
                    for j, network in enumerate(networks):

                        history, metrics = run_neural_network(
                            res, images, cities, epoch_range, lr, network)

                        results.loc[rowId] = ['Netzwerk ' + str(j+1), images, res, lr, history.history['accuracy'],
                                              history.history['val_accuracy'], history.history['loss'],
                                              history.history['val_loss'], metrics]

                        rowId = rowId + 1
    except Exception as e:
        logging.error(traceback.format_exc())
    try:
        results.to_excel(resultPath)
    except Exception as e:
        print("Didn't write to file!")
        print(e)
    return results  # safe path if write doesn't work


main()