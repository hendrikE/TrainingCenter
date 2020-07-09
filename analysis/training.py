import os
import pickle
import time

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def load_features(split, feature_path):
    """
    Function to load features, split and shuffle them
    :param split: Tuple with the amount of training and test data, e.g. (0.8, 0.2)
    :param feature_path: Path to the directory containing the NPY files with the features
    :return: List consisting of one array for the training data and one for the test data,
    List consisting of one array for the training labels and one for the test labels
    """
    distributions = os.listdir(feature_path)

    # train, val, test
    data = [[], []]
    labels = [[], []]

    for distribution in distributions:
        cls_data = np.load(os.path.join(feature_path, distribution))
        length = cls_data.shape[0]
        data[0].append(cls_data[:int(length * split[0])])
        data[1].append(cls_data[int(length * split[0]):])
        for i in range(2):
            labels[i].append(np.full((int(length * split[i]), 1), int(distribution.split(".")[0].split("_")[1]) - 1))

    data_merged = []
    labels_merged = []

    for i, d in enumerate(data):
        data_merged.append(np.vstack(d))
        labels_merged.append(np.array(labels[i]).flatten())

    data_shuffled = []
    labels_shuffled = []

    for i, d in enumerate(data_merged):
        indices = np.arange(d.shape[0])
        np.random.shuffle(indices)
        data_shuffled.append(d[indices])
        labels_shuffled.append(labels_merged[i][indices])

    return data_shuffled, labels_shuffled


def features_train_test(classifiers, train_data, train_label, test_data, test_label):
    """
    Function to train and test different classifiers
    :param classifiers: List with classifiers
    :param train_data: Array with features for training
    :param train_label: Array with labels for training
    :param test_data: Array with features for testing
    :param test_label: Array with labels for testing
    :return: List consisting of confusion matrix arrays, List of accuracies
    """
    confusion_matrices = []
    accuracies = []
    for classifier in classifiers:
        classifier.fit(train_data, train_label)
        confusion_matrices.append(confusion_matrix(test_label, classifier.predict(test_data)))
        accuracies.append(classifier.score(test_data, test_label))

    return confusion_matrices, accuracies


def feature_training(feature_path, results_path, seg_name, train_split=90,
                     cls_set="basic", save_confusion_matrices=True):
    """
    Function to run the complete training pipeline
    :param feature_path: Path to the directory containing the NPY files with the features
    :param results_path: Path to the directory where the results should be saved
    :param seg_name: Name of the segmentation
    :param train_split: Percentage of data used for training, e.g. 80
    :param cls_set: Either a list of names of classifiers to be used, or a the single name of classifier; if not used
    all classifiers will be used
    :param save_confusion_matrices: True if the confusion matrices should be used
    :return: List of accuracies
    """
    split = (train_split / 100, (100 - train_split) / 100)
    (train_data, test_data), (train_label, test_label) = load_features(split, feature_path)

    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes", "QDA"]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]

    if type(cls_set) == list:
        if all(x in names for x in cls_set):
            classifiers = [classifiers[names.index(classifier)] for classifier in cls_set]
            names = cls_set
    elif cls_set in names:
        classifiers = [classifiers[names.index(cls_set)]]
        names = [cls_set]

    confusion_matrices, accuracies = features_train_test(classifiers,
                                                         train_data, train_label,
                                                         test_data, test_label)

    accuracies = dict(zip(names, accuracies))

    if save_confusion_matrices:
        confusion_matrices = dict(zip(names, confusion_matrices))
        with open(os.path.join(results_path, "confusion_matrices_seg_{}".format(seg_name)), "wb") as matrix_file:
            pickle.dump(confusion_matrices, matrix_file)

    return accuracies


def feature_training_duration(feature_path, train_split=90):
    """
    Function to compare how long it takes to train on the data, create a confusion matrix and calculate the accuracy
    for each classifier
    :param feature_path: Path to the directory containing the NPY files with the features
    :param train_split: Percentage of data used for training, e.g. 80
    :return:
    """
    split = (train_split / 100, (100 - train_split) / 100)
    (train_data, test_data), (train_label, test_label) = load_features(split, feature_path)
    print("Loaded Features")

    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes", "QDA"]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]

    confusion_matrices = []
    accuracies = []
    for name, classifier in zip(names, classifiers):
        start_time = time.time()
        classifier.fit(train_data, train_label)
        train_time = time.time()
        confusion_matrices.append(confusion_matrix(test_label, classifier.predict(test_data)))
        matrix_time = time.time()
        accuracies.append(classifier.score(test_data, test_label))
        acc_time = time.time()

        print(">>>> Classifier {} used {}s for training, "
              "{}s for creating the matrix "
              "and {}s for calculating the accuracy.".format(name,
                                                             train_time - start_time,
                                                             matrix_time - train_time,
                                                             acc_time - matrix_time))


def train_and_save_model(feature_path, model_path, classifier, train_split=90):
    """
    Function to train a model and save it together with test data
    :param feature_path: Path to the directory containing the NPY files with the features
    :param model_path: Path to the directory where the model and test data will be stored
    :param classifier: Name of classifier to be used
    :param train_split: Percentage of data used for training, e.g. 80
    :return:
    """
    split = (train_split / 100, (100 - train_split) / 100)
    (train_data, test_data), (train_label, test_label) = load_features(split, feature_path)

    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes", "QDA"]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]

    classifier = classifiers[names.index(classifier)]
    classifier.fit(train_data, train_label)

    with open(os.path.join(model_path, "model"), "wb") as file:
        pickle.dump(classifier, file)

    with open(os.path.join(model_path, "test_data"), "wb") as file:
        pickle.dump((test_data, test_label), file)
