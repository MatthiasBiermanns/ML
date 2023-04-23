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

base_dir = "/home/fhase/Desktop/Dtaset_4000"
batch_size = 1


def get_class_names():
    data_dir = pathlib.Path(base_dir).with_suffix("")
    class_names = np.array(sorted([item.name for item in data_dir.glob(
        '*') if item.name != "NeuralNetwork_Results.xlsx"]))
    return class_names


def get_data(res):
    x = np.random.randint(0, 100)
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        base_dir,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=32,
        image_size=res,
        shuffle=True,
        seed=x,
        validation_split=0.2,
        subset="training",
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        base_dir,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=32,
        image_size=res,
        shuffle=True,
        seed=x,
        validation_split=0.2,
        subset="validation",
    )
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


def run_neural_network(res, labels, epoch_range, learning_rate, layers):

    train_ds, val_ds = get_data(
        res)

    model = generate_model(layers)
    print(model.summary())

    model, history = train_model(
        model, learning_rate, train_ds, val_ds, epoch_range)

    test_ds = get_test_data(image_size=res, test_size=50)
    x_val = list(map(lambda x: x[0].numpy().tolist(), test_ds))
    y_val = list(map(lambda x: x[1].numpy(), test_ds))
    metrics = get_metrics(model, labels, x_val, y_val)
    print(metrics)

    return history, metrics


def main():
    resultPath = os.path.join(base_dir, 'NeuralNetwork_Results.xlsx')
    cities = get_class_names()
    resolution = [(720, 480)]
    learning_rate = [0.00001]

    results = pd.DataFrame(columns=['ID', 'Netzwerk', 'Bildauflösung (px)', 'Lernquote',
                                    'Train Acc', 'Val Acc', 'Train Loss', 'Val Loss', 'Metriken'])
    results = results.set_index('ID')
    rowId = 0
    try:
        for res in resolution:
            for lr in learning_rate:
                network1 = [
                    Conv2D(16, 3, padding="same", activation="tanh",
                           input_shape=(res[0], res[1], 3)),
                    Conv2D(32, 3, padding="same", activation="tanh"),
                    MaxPool2D(),
                    Conv2D(64, 3, padding="same",
                           activation="tanh"),
                    MaxPool2D(),
                    Dropout(0.4),
                    Flatten(),
                    Dense(128, activation="tanh"),
                    Dense(len(cities), activation="softmax")
                ]
                network2 = [
                    Conv2D(32, 3, padding="same", activation="relu",
                           input_shape=(res[0], res[1], 3)),
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
                networks = [network1,network2]
                for j, network in enumerate(networks):
                    history, metrics = run_neural_network(
                        res, cities, 15 if j == 0 else 22, lr, network)

                    results.loc[rowId] = ['Netzwerk ' + str(j+1), res, lr, history.history['accuracy'],
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
