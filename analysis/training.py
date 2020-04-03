# TODO: improve simple training script, introduce time-series analysis
import os

import keras
from keras import backend
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout
# from keras.utils import to_categorical

batch_size = 100
no_epochs = 30
learning_rate = 0.001
validation_split = 0.2
verbosity = 1


def load_data():
    return []


def data_generator(files):
    while True:
        grids = []
        labels = []


def build_model(no_classes, sample_shape):
    model = Sequential()
    model.add(
        Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=sample_shape))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(no_classes, activation='softmax'))

    return model


def training(model, data):
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=learning_rate),
                  metrics=['accuracy'])

    history = model.fit(data["x_test"], data["y_test"],
                        batch_size=batch_size,
                        epochs=no_epochs,
                        verbose=verbosity,
                        validation_split=validation_split)

    return history


def run(no_classes, sample_shape):
    data = os.listdir(os.path.join("analysis_files", "samples"))
    model = build_model(no_classes, sample_shape)
    results = training(model, data)
    backend.clear_session()

    return results
