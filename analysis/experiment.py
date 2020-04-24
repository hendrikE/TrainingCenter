# TODO: create an experimental setup that samples data with different parameters, train on it and compare results
# standard python libraries
import os
import json
import pickle

# installed python libraries
import numpy as np

# parts of TrainingCenter
from analysis import sampling, training, visualize


def showcase_example():
    pass


def create_class_distributions(distribution_sets=None, size=500):
    """
    This function creates for existing distribution parameters different distribution classes
    :param distribution_sets: Which distribution set to create classes for, leave None if all should be used
    :param size: How many distributions per class
    :return:
    """
    if distribution_sets is None:
        distribution_sets = \
            [x for x in os.listdir(os.path.join("analysis_files", "distributions")) if x.endswith(".json")]
    for distribution_set in distribution_sets:
        with open(os.path.join("analysis_files", "distributions", distribution_set), "r+") as class_file:
            classes = json.load(class_file)
        file_path = os.path.join("analysis_files", "distributions", distribution_set.split(".")[0])
        os.mkdir(file_path)
        for cls, val in classes.items():
            sampling.create_distributions(file_path, cls, val, size=size, save=True)


def show_distributions(env, distribution_sets=None):
    if distribution_sets is None:
        distribution_sets = \
            [x for x in os.listdir(os.path.join("analysis_files", "distributions")) if x.endswith(".json")]
    env_path = os.path.join("analysis_files", "environments", "{}.json".format(env))
    for distribution_set in distribution_sets:
        distribution_set_path = os.path.join("analysis_files", "distributions", "{}.json".format(distribution_set))
        visualize.visualize_distributions(env_path, distribution_set_path)


def create_time_series_distributions():
    pass


def create_segmentations(env, segmentation_sets=None):
    """
    Create segmentations consisting of positions and coordinates based on existing parameters
    :param env: The environment which should be segmented
    :param segmentation_sets: Which segmentations to use
    :return:
    """
    if segmentation_sets is None:
        segmentation_sets = \
            [x for x in os.listdir(os.path.join("analysis_files", "segmentations")) if x.endswith(".json")]
    with open(os.path.join("analysis_files", "environments", "{}.json".format(env)), "r+") as env_file:
        environment = json.load(env_file)
    if environment["basic_shape"] == "cuboid":
        for el in segmentation_sets:
            with open(os.path.join("analysis_files", "segmentations", el), "r+") as seg_file:
                segmentations = json.load(seg_file)
            file_path = os.path.join("analysis_files", "segmentations", "{}_ENV_{}".format(el.split(".")[0], env))
            os.mkdir(file_path)
            for cls, val in segmentations.items():
                sampling.cuboid_segmentation(file_path, environment, cls, val, save=True)


def show_segmentation():
    pass


def draw_samples(segmentation_sets=None, distribution_sets=None):
    """
    This function allows to draw samples from distributions based on given segmentations.
    :param segmentation_sets: A list of segmentation sets that should be used, leave None if all should be used
    :param distribution_sets: A list of distribution sets that should be used, leave None if all should be used
    :return:
    """
    if segmentation_sets is None:
        segmentation_sets = \
            [x for x in os.listdir(os.path.join("analysis_files", "segmentations")) if not x.endswith(".json")]
    if distribution_sets is None:
        distribution_sets = \
            [x for x in os.listdir(os.path.join("analysis_files", "distributions")) if not x.endswith(".json")]

    for segmentation_set in segmentation_sets:
        segmentation_set_path = os.path.join("analysis_files", "samples", segmentation_set)
        os.mkdir(segmentation_set_path)
        segmentations = os.listdir(os.path.join("analysis_files", "segmentations", segmentation_set))

        for distribution_set in distribution_sets:
            distribution_set_path = os.path.join(segmentation_set_path, distribution_set)
            os.mkdir(distribution_set_path)
            distributions = os.listdir(os.path.join("analysis_files", "distributions", distribution_set))

            for segmentation in segmentations:
                segmentation_loaded = np.load(os.path.join("analysis_files", "segmentations",
                                                           segmentation_set, segmentation))
                length = max(segmentation_loaded[:, 0]) + 1
                width = max(segmentation_loaded[:, 1]) + 1
                height = max(segmentation_loaded[:, 2]) + 1
                size = int(length), int(width), int(height)

                segmentation_path = os.path.join(distribution_set_path,
                                                 "segmentation_{}".format(segmentation.split(".")[0]))
                os.mkdir(segmentation_path)

                for distribution in distributions:
                    distribution_elements = np.load(os.path.join("analysis_files", "distributions",
                                                                 distribution_set, distribution))

                    distribution_path = os.path.join(segmentation_path,
                                                     "distribution_{}".format(distribution.split(".")[0]))
                    os.mkdir(distribution_path)

                    for index, element in enumerate(distribution_elements):
                        element_path = os.path.join(distribution_path, str(index))
                        sampling.draw_sample(element, segmentation_loaded, element_path, size)


def show_overlay():
    pass


def show_samples():
    pass


def run_training(sampling_on_the_fly, segmentation_sets=None, distribution_sets=None):
    """
    This function allows to train a model for contamination detection.
    :param sampling_on_the_fly: True, if artificial data should be used and sampled during training (as opposed to
    using already existing samples)
    :param segmentation_sets: Which segmentation sets should be used, leave None if all should be used
    :param distribution_sets: Which distribution sets should be used, leave None if all should be used
    :return:
    """
    if segmentation_sets is None:
        segmentation_sets = \
            [x for x in os.listdir(os.path.join("analysis_files", "segmentations")) if not x.endswith(".json")]
    if distribution_sets is None:
        distribution_sets = \
            [x for x in os.listdir(os.path.join("analysis_files", "distributions")) if not x.endswith(".json")]

    for segmentation_set in segmentation_sets:
        segmentation_set_path = os.path.join("analysis_files", "results", segmentation_set)
        os.mkdir(segmentation_set_path)
        segmentations = os.listdir(os.path.join("analysis_files", "segmentations", segmentation_set))
        segmentations.sort()

        for distribution_set in distribution_sets:
            distribution_set_path = os.path.join(segmentation_set_path, distribution_set)
            os.mkdir(distribution_set_path)

            if sampling_on_the_fly:
                accuracies = {}
                for segmentation in segmentations:
                    segmentation_loaded = np.load(os.path.join("analysis_files", "segmentations",
                                                               segmentation_set, segmentation))
                    segmentation_path = os.path.join(distribution_set_path,
                                                     "segmentation_{}".format(segmentation.split(".")[0]))
                    os.mkdir(segmentation_path)
                    segmentation_name = segmentation.split(".")[0]
                    accuracy = training.train_with_sampling_on_the_fly(
                        segmentation_loaded,
                        segmentation_path,
                        distribution_set
                    )
                    accuracies[segmentation_name] = accuracy

                with open(os.path.join(distribution_set_path, "accuracies.json"), "wb") as acc_file:
                    pickle.dump(accuracies, acc_file)

            else:
                # TODO implement training with already drawn samples
                pass


def show_training_results():
    pass
