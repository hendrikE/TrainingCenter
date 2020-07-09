import os
import json
import pickle

import numpy as np

from analysis import sampling, training, visualize, features


def create_class_distributions(distribution_sets=None, size=500):
    """
    Function to create distributions belonging to different classes for different distribution sets
    :param distribution_sets: List of JSON file names containing the different distribution set specifications
    :param size: Number of distributions per class to be created
    :return:
    """
    if distribution_sets is None:
        distribution_sets = \
            [x.split(".")[0] for x in os.listdir(os.path.join("analysis_files", "distributions"))
             if x.endswith(".json")]
    for distribution_set in distribution_sets:
        with open(os.path.join("analysis_files", "distributions", "{}.json".format(distribution_set)), "r+") as class_file:
            classes = json.load(class_file)
        file_path = os.path.join("analysis_files", "distributions", distribution_set)
        os.mkdir(file_path)
        for cls, val in classes.items():
            sampling.create_distributions(val, size=size, save=True, file_path=file_path, dist_class=cls)


def show_distributions(env, distribution_sets=None):
    """
    Function to visualize the different distribution sets with their respective classes
    :param env: JSON file name containing information on the environment
    :param distribution_sets: List of JSON file names containing the different distribution set specifications
    :return:
    """
    if distribution_sets is None:
        distribution_sets = \
            [x.split(".")[0] for x in os.listdir(os.path.join("analysis_files", "distributions"))
             if x.endswith(".json")]
    env_path = os.path.join("analysis_files", "environments", "{}.json".format(env))
    for distribution_set in distribution_sets:
        distribution_set_path = os.path.join("analysis_files", "distributions", "{}.json".format(distribution_set))
        visualize.visualize_distributions(env_path, distribution_set_path, distribution_set)


def create_segmentations(env, segmentation_sets=None):
    """
    Function to create segmentations of a given environment
    :param env: JSON file name containing information on the environment
    :param segmentation_sets: List of JSON file names containing the different segmentation set specifications
    :return:
    """
    if segmentation_sets is None:
        segmentation_sets = \
            [x.split(".")[0] for x in os.listdir(os.path.join("analysis_files", "segmentations")) if x.endswith(".json")]
    with open(os.path.join("analysis_files", "environments", "{}.json".format(env)), "r+") as env_file:
        environment = json.load(env_file)
    if environment["basic_shape"] == "cuboid":
        for segmentation_set in segmentation_sets:
            with open(os.path.join("analysis_files", "segmentations", "{}.json".format(segmentation_set)), "r+") as seg_file:
                segmentations = json.load(seg_file)
            file_path = os.path.join("analysis_files",
                                     "segmentations",
                                     "{}_ENV_{}".format(segmentation_set, env))
            os.mkdir(file_path)
            for cls, val in segmentations.items():
                sampling.create_cuboid_segmentation(environment, val,
                                                    save=True, file_path=file_path, segmentation_class=cls)


def show_segmentation(environment, segmentation_sets=None):
    """
    Function to visualize the different segmentation sets
    :param environment: JSON file name containing information on the environment
    :param segmentation_sets: List of the directories, where the segmentation sets have been saved after creation
    :return:
    """
    if segmentation_sets is None:
        segmentation_sets = \
            [x for x in os.listdir(os.path.join("analysis_files", "segmentations")) if not x.endswith(".json")]
    env_path = os.path.join("analysis_files", "environments", "{}.json".format(environment))
    for segmentation_set in segmentation_sets:
        segmentation_set_path = os.path.join("analysis_files", "segmentations", segmentation_set)
        visualize.visualize_segmentations(env_path, segmentation_set_path, segmentation_set)


def create_features(segmentation_sets=None, distribution_sets=None):
    """
    Function to create features from distributions based on given segmentations
    :param segmentation_sets: A list of segmentation set directories
    :param distribution_sets: A list of distribution set directories
    :return:
    """
    if segmentation_sets is None:
        segmentation_sets = \
            [x for x in os.listdir(os.path.join("analysis_files", "segmentations")) if not x.endswith(".json")]
    if distribution_sets is None:
        distribution_sets = \
            [x for x in os.listdir(os.path.join("analysis_files", "distributions")) if not x.endswith(".json")]

    for segmentation_set in segmentation_sets:
        segmentation_set_path = os.path.join("analysis_files", "features", segmentation_set)
        if not os.path.exists(segmentation_set_path):
            os.mkdir(segmentation_set_path)
        segmentations = os.listdir(os.path.join("analysis_files", "segmentations", segmentation_set))

        for segmentation in segmentations:
            segmentation_loaded = np.load(os.path.join("analysis_files", "segmentations",
                                                       segmentation_set, segmentation))
            length = max(segmentation_loaded[:, 0]) + 1
            width = max(segmentation_loaded[:, 1]) + 1
            height = max(segmentation_loaded[:, 2]) + 1
            size = int(length), int(width), int(height)

            segmentation_path = os.path.join(segmentation_set_path,
                                             "segmentation_{}".format(segmentation.split(".")[0]))
            os.mkdir(segmentation_path)

            for distribution_set in distribution_sets:
                distribution_set_path = os.path.join(segmentation_path, distribution_set)
                os.mkdir(distribution_set_path)
                distributions = os.listdir(os.path.join("analysis_files", "distributions", distribution_set))

                for distribution in distributions:
                    distribution_elements = np.load(os.path.join("analysis_files", "distributions",
                                                                 distribution_set, distribution))

                    distribution_path = os.path.join(distribution_set_path,
                                                     "distribution_{}".format(distribution.split(".")[0]))
                    features.generate_features(distribution_elements, segmentation_loaded, size,
                                               save=True, path=distribution_path)


def create_modified_features(segmentation_sets=None, distribution_sets=None,
                             deviation_set=None, incompleteness_set=None):
    """
    Function to create features from distributions based on given segmentations with modifications
    :param incompleteness_set: List with probabilities for each data point to be missing
    :param deviation_set: List of variances by which the data points are supposed to deviate from basic segmentation
    :param segmentation_sets: A list of segmentation set directories
    :param distribution_sets: A list of distribution set directories
    :return:
    """
    assert (deviation_set is not None or incompleteness_set is not None) and \
           (deviation_set is None or incompleteness_set is None), "Either Deviation or Incompleteness should be given"

    if segmentation_sets is None:
        segmentation_sets = \
            [x for x in os.listdir(os.path.join("analysis_files", "segmentations")) if not x.endswith(".json")]
    if distribution_sets is None:
        distribution_sets = \
            [x for x in os.listdir(os.path.join("analysis_files", "distributions")) if not x.endswith(".json")]

    if deviation_set is not None:
        for deviation in deviation_set:
            for segmentation_set in segmentation_sets:
                segmentation_set_path = os.path.join("analysis_files", "features",
                                                     "{}_INC_{}_DEV_{}".format(segmentation_set,
                                                                               0, deviation))
                if not os.path.exists(segmentation_set_path):
                    os.mkdir(segmentation_set_path)
                segmentations = os.listdir(os.path.join("analysis_files", "segmentations", segmentation_set))

                for segmentation in segmentations:
                    segmentation_loaded = np.load(os.path.join("analysis_files", "segmentations",
                                                               segmentation_set, segmentation))
                    length = max(segmentation_loaded[:, 0]) + 1
                    width = max(segmentation_loaded[:, 1]) + 1
                    height = max(segmentation_loaded[:, 2]) + 1
                    size = int(length), int(width), int(height)

                    segmentation_path = os.path.join(segmentation_set_path,
                                                     "segmentation_{}".format(segmentation.split(".")[0]))
                    os.mkdir(segmentation_path)

                    for distribution_set in distribution_sets:
                        distribution_set_path = os.path.join(segmentation_path, distribution_set)
                        os.mkdir(distribution_set_path)
                        distributions = os.listdir(os.path.join("analysis_files", "distributions", distribution_set))

                        for distribution in distributions:
                            distribution_elements = np.load(os.path.join("analysis_files", "distributions",
                                                                         distribution_set, distribution))

                            distribution_path = os.path.join(distribution_set_path,
                                                             "distribution_{}".format(distribution.split(".")[0]))
                            features.generate_features_with_deviation(distribution_elements,
                                                                      segmentation_loaded, size,
                                                                      deviation,
                                                                      save=True, path=distribution_path)
    else:
        for incompleteness in incompleteness_set:
            for segmentation_set in segmentation_sets:
                segmentation_set_path = os.path.join("analysis_files", "features",
                                                     "{}_INC_{}_DEV_{}".format(segmentation_set,
                                                                               incompleteness * 100, 0))
                if not os.path.exists(segmentation_set_path):
                    os.mkdir(segmentation_set_path)
                segmentations = os.listdir(os.path.join("analysis_files", "segmentations", segmentation_set))

                for segmentation in segmentations:
                    segmentation_loaded = np.load(os.path.join("analysis_files", "segmentations",
                                                               segmentation_set, segmentation))
                    length = max(segmentation_loaded[:, 0]) + 1
                    width = max(segmentation_loaded[:, 1]) + 1
                    height = max(segmentation_loaded[:, 2]) + 1
                    size = int(length), int(width), int(height)

                    segmentation_path = os.path.join(segmentation_set_path,
                                                     "segmentation_{}".format(segmentation.split(".")[0]))
                    os.mkdir(segmentation_path)

                    for distribution_set in distribution_sets:
                        distribution_set_path = os.path.join(segmentation_path, distribution_set)
                        os.mkdir(distribution_set_path)
                        distributions = os.listdir(os.path.join("analysis_files", "distributions", distribution_set))

                        for distribution in distributions:
                            distribution_elements = np.load(os.path.join("analysis_files", "distributions",
                                                                         distribution_set, distribution))

                            distribution_path = os.path.join(distribution_set_path,
                                                             "distribution_{}".format(distribution.split(".")[0]))
                            features.generate_features_with_incompleteness(distribution_elements,
                                                                           segmentation_loaded, size,
                                                                           incompleteness,
                                                                           save=True, path=distribution_path)


def show_features(environment, segmentation_sets, distribution_sets):
    """
    Function to create visualizations for different segmentation and distribution sets
    :param environment: JSON file name containing information on the environment
    :param segmentation_sets: A list of segmentation set directories
    :param distribution_sets: A list of JSON file names containing the distribution set specifications
    :return:
    """
    env_path = os.path.join("analysis_files", "environments", "{}.json".format(environment))
    for segmentation_set in segmentation_sets:
        segmentation_set_path = os.path.join("analysis_files", "segmentations", segmentation_set)
        for segmentation in os.listdir(segmentation_set_path):
            segmentation_loaded = np.load(os.path.join("analysis_files", "segmentations",
                                                       segmentation_set, segmentation))
            length = max(segmentation_loaded[:, 0]) + 1
            width = max(segmentation_loaded[:, 1]) + 1
            height = max(segmentation_loaded[:, 2]) + 1
            size = int(length), int(width), int(height)
            for distribution_set in distribution_sets:
                distribution_set_path = os.path.join("analysis_files",
                                                     "distributions",
                                                     "{}.json".format(distribution_set))
                with open(distribution_set_path, "r+") as distribution_file:
                    distribution_set_loaded = json.load(distribution_file)
                distributions = [([((x[0] + x[1]) / 2) for x in val], key)
                                 for key, val in distribution_set_loaded.items()]
                for distribution in distributions:
                    visualize.visualize_feature_creation(distribution_set, distribution[1], distribution[0],
                                                         segmentation_set, segmentation.split(".")[0],
                                                         segmentation_loaded, size, env_path)


def run_feature_training(segmentation_sets=None, distribution_sets=None, cls_set="basic"):
    """
    Function to train a model for classification with multiple segmentation and distribution sets
    :param segmentation_sets: A list of segmentation set directories
    :param distribution_sets: A list of distribution set directories
    :param cls_set: List of classifiers, or name of a single classifier
    :return:
    """
    if segmentation_sets is None:
        segmentation_sets = \
            [x for x in os.listdir(os.path.join("analysis_files", "segmentations")) if not x.endswith(".json")]
    if distribution_sets is None:
        distribution_sets = \
            [x for x in os.listdir(os.path.join("analysis_files", "distributions")) if not x.endswith(".json")]

    for segmentation_set in segmentation_sets:
        segmentation_set_path = os.path.join("analysis_files", "feature_results", segmentation_set)
        if not os.path.exists(segmentation_set_path):
            os.mkdir(segmentation_set_path)
        segmentations = os.listdir(os.path.join("analysis_files", "segmentations", segmentation_set))
        segmentations.sort()

        for distribution_set in distribution_sets:
            distribution_set_path = os.path.join(segmentation_set_path, distribution_set)
            os.mkdir(distribution_set_path)
            accuracies = {}
            for segmentation in segmentations:
                segmentation_name = segmentation.split(".")[0]
                feature_path = os.path.join("analysis_files", "features",
                                            segmentation_set,
                                            "segmentation_{}".format(segmentation.split(".")[0]),
                                            distribution_set)
                print("Training on Segmentation {}".format(segmentation))
                accuracy = training.feature_training(
                    feature_path,
                    distribution_set_path,
                    segmentation_name,
                    cls_set=cls_set
                )
                accuracies[segmentation_name] = accuracy

            with open(os.path.join(distribution_set_path, "accuracies"), "wb") as acc_file:
                pickle.dump(accuracies, acc_file)


def show_feature_training_results(segmentation_sets=None):
    """
    Function to show the results of training on basic data
    :param segmentation_sets: A list of segmentation set directories
    :return:
    """
    for segmentation_set in segmentation_sets:
        path = os.path.join("analysis_files", "feature_results", segmentation_set)
        distribution_sets = os.listdir(path)
        visualize.visualize_feature_accuracies([os.path.join(path, dist_set) for dist_set in distribution_sets])


def run_feature_training_with_variations(segmentation_sets=None, single_segmentations=None,
                                         distribution_sets=None,
                                         cls_set="basic", train_splits=None,
                                         save_confusion_matrices=False):
    """
    Function to train a model for classification with multiple segmentation and distribution sets with variables
    :param segmentation_sets: A list of segmentation set directories as they can be found in the "features" directory
    :param single_segmentations: A list of lists containing single segmentations from each segmentation set
    :param distribution_sets: A list of distribution set directories
    :param cls_set: List of classifiers, or name of a single classifier
    :param train_splits: List of percentages of much of the data is to be used for training
    :param save_confusion_matrices: True if matrices should be saved
    :return:
    """
    if train_splits is None:
        train_splits = [90]

    if segmentation_sets is None:
        segmentation_sets = \
            [x for x in os.listdir(os.path.join("analysis_files", "segmentations")) if not x.endswith(".json")]
    if distribution_sets is None:
        distribution_sets = \
            [x for x in os.listdir(os.path.join("analysis_files", "distributions")) if not x.endswith(".json")]

    for train_split in train_splits:
        print("Training on Split '{}'".format(train_split))
        for seg_index, segmentation_set in enumerate(segmentation_sets):
            print("> Training Segmentation Set '{}'".format(segmentation_set))
            segmentation_set_path = os.path.join("analysis_files", "feature_results",
                                                 "TS_{}_SEG_".format(train_split) + segmentation_set)
            if not os.path.exists(segmentation_set_path):
                os.mkdir(segmentation_set_path)
            if single_segmentations is None:
                segmentations = os.listdir(os.path.join("analysis_files", "segmentations", segmentation_set))
                segmentations.sort()
            else:
                segmentations = single_segmentations[seg_index]

            for distribution_set in distribution_sets:
                print(">> Training on Distribution Set '{}'".format(distribution_set))
                distribution_set_path = os.path.join(segmentation_set_path, distribution_set)
                os.mkdir(distribution_set_path)
                accuracies = {}
                for segmentation in segmentations:
                    segmentation_name = segmentation.split(".")[0]
                    feature_path = os.path.join("analysis_files", "features",
                                                segmentation_set,
                                                "segmentation_{}".format(segmentation.split(".")[0]),
                                                distribution_set)
                    print(">>> Training on Segmentation '{}'".format(segmentation))
                    accuracy = training.feature_training(
                        feature_path,
                        distribution_set_path,
                        segmentation_name,
                        cls_set=cls_set,
                        train_split=train_split,
                        save_confusion_matrices=save_confusion_matrices
                    )
                    accuracies[segmentation_name] = accuracy

                with open(os.path.join(distribution_set_path, "accuracies"), "wb") as acc_file:
                    pickle.dump(accuracies, acc_file)


def test_duration_feature_training():
    """
    Function to compare training duration for different classifiers on the given data
    :return:
    """
    distribution_sets = ["combined_classes", "orientation_classes", "position_classes", "shape_classes"]

    for distribution_set in distribution_sets:
        print(">>> Training on Distribution Set {}".format(distribution_set))
        feature_path = os.path.join("analysis_files", "features",
                                    "basic_cuboid_ENV_basic_INC_0_DEV_0",
                                    "segmentation_5_5_5",
                                    distribution_set)
        training.feature_training_duration(feature_path)


def train_and_save_model():
    feature_path = os.path.join("analysis_files", "features", "basic_cuboid_ENV_basic_INC_0_DEV_0",
                                "segmentation_5_5_5", "combined_classes")
    model_path = "mock_setup"
    classifier = "Nearest Neighbors"
    training.train_and_save_model(feature_path, model_path, classifier)
