# TODO: introduce time-series analysis
import os
import re
import pickle

import numpy as np
import keras
from keras import backend
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.stats import multivariate_normal

batch_size = 10
no_epochs = 5
learning_rate = 0.001
validation_split = 0.2
verbosity = 1


def load_data(split, distribution_set):
    classes = os.listdir(os.path.join("analysis_files", "distributions", distribution_set))
    no_classes = len(classes)

    # train, val, test
    data = [[], [], []]
    labels = [[], [], []]

    for cls in classes:
        cls_data = np.load(os.path.join("analysis_files", "distributions", distribution_set, cls))
        length = cls_data.shape[0]
        data[0].append(cls_data[:int(length * split[0])])
        data[1].append(cls_data[int(length * split[0]):int(length * (split[1] + split[0]))])
        data[2].append(cls_data[int(length * (split[1] + split[0])):])
        for i in range(3):
            labels[i].append(np.full((int(length * split[i]), 1), int(cls.split(".")[0]) - 1))

    data_merged = []
    labels_merged = []

    for i, d in enumerate(data):
        data_merged.append(np.vstack(d))
        labels_merged.append(np.vstack(labels[i]))

    data_shuffled = []
    labels_shuffled = []

    for i, d in enumerate(data_merged):
        indices = np.arange(d.shape[0])
        np.random.shuffle(indices)
        data_shuffled.append(d[indices])
        labels_shuffled.append(labels_merged[i][indices])

    categorical_labels = []

    for lab in labels_shuffled:
        categorical_labels.append(to_categorical(lab))

    return data_shuffled, categorical_labels, no_classes


def data_generator_sampling_on_the_fly(data, labels, seg, size, test_mode):
    indices = np.arange(data.shape[0])
    batch = []
    while True:
        if not test_mode:
            np.random.shuffle(indices)
        for i in indices:
            batch.append(i)
            if len(batch) == batch_size:
                grids = []
                for b in batch:
                    dist = multivariate_normal(data[b][:3], np.diag(data[b][3:]))
                    grid = np.zeros((size[0], size[1], size[2]))
                    sample = dist.pdf(seg[:, 3:])
                    for index, s in enumerate(seg[:, :3]):
                        l, w, h = s
                        grid[int(l), int(w), int(h)] = sample[index]
                    grids.append(np.expand_dims(grid, axis=3))
                grids = np.array(grids)
                yield grids, labels[batch]
                batch = []


def data_generator_w_sampling_on_the_fly(data, labels, seg, test_mode):
    while True:
        grids = []
        labels = []


def build_model(no_classes, sample_shape):
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=sample_shape+(1,)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(no_classes, activation='softmax'))

    return model


def training(model, train_gen, val_gen, no_train_data, no_val_data):
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=learning_rate),
                  metrics=['accuracy'])

    history = model.fit_generator(
        train_gen,
        steps_per_epoch=no_train_data // batch_size,
        validation_data=val_gen,
        validation_steps=no_val_data // batch_size,
        epochs=no_epochs,
        verbose=verbosity
    )

    return history, model


def test(test_gen, model, no_test_data):
    prediction = model.predict_generator(
        test_gen,
        steps=no_test_data // batch_size
    )
    prediction = np.argmax(prediction, axis=1)
    return prediction


def train_with_sampling_on_the_fly(segmentation,
                                   segmentation_loaded,
                                   segmentation_results_path,
                                   segmentation_models_path,
                                   distribution_set
                                   ):

    (train_data, val_data, test_data), (train_label, val_label, test_label), no_classes = load_data((0.8, 0.1, 0.1),
                                                                                                    distribution_set)
    print("Loaded Data.")

    size = int(max(segmentation_loaded[:, 0]) + 1), int(max(segmentation_loaded[:, 1]) + 1), int(max(segmentation_loaded[:, 2]) + 1)

    model = build_model(no_classes, size)
    print("Build Model.")

    training_generator = data_generator_sampling_on_the_fly(train_data, train_label, segmentation_loaded, size, False)
    validation_generator = data_generator_sampling_on_the_fly(val_data, val_label, segmentation_loaded, size, False)
    results, model = training(model,
                              training_generator,
                              validation_generator,
                              train_data.shape[0],
                              val_data.shape[0])
    print("Finished Training.")

    test_generator = data_generator_sampling_on_the_fly(test_data, test_label, segmentation_loaded, size, True)
    test_results = test(test_generator, model, test_data.shape[0])
    test_label = np.argmax(test_label, axis=1)
    prediction_results = np.stack((test_label, test_results), axis=-1)
    conf_matrix = confusion_matrix(test_label, test_results)
    print("Finished Testing.")

    model.save(os.path.join(segmentation_models_path, "model.h5"))
    plot_model(model, to_file=os.path.join(segmentation_models_path, "model.png"))

    with open(os.path.join(segmentation_results_path, "history"), "wb") as results_file:
        pickle.dump(results.history, results_file)
    np.save(os.path.join(segmentation_results_path, "prediction_results"), prediction_results)
    np.save(os.path.join(segmentation_results_path, "confusion_matrix"), conf_matrix)

    backend.clear_session()

    return accuracy_score(test_label, test_results)


def train_without_sampling_on_the_fly():
    data = os.listdir(os.path.join("analysis_files", "samples"))
