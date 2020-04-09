# TODO: create an experimental setup that samples data with different parameters, train on it and compare results
# standard python libraries
import os
import json
import pickle

# installed python libraries
import numpy as np

# parts of TrainingCenter
from analysis import sampling, training


def showcase_example():
    pass


def create_class_distributions(all_el=True, elements=None):
    if all_el:
        elements = [x for x in os.listdir(os.path.join("analysis_files", "distributions")) if x.endswith(".json")]
    for el in elements:
        with open(os.path.join("analysis_files", "distributions", el), "r+") as class_file:
            classes = json.load(class_file)

        file_path = os.path.join("analysis_files", "distributions", el.split(".")[0])
        os.mkdir(file_path)

        for cls, val in classes.items():
            sampling.create_distributions(file_path, cls, val, size=500, save=True)


def create_time_series_distributions():
    pass


def create_segmentations(env, all_el=True, elements=None):
    if all_el:
        elements = [x for x in os.listdir(os.path.join("analysis_files", "segmentations")) if x.endswith(".json")]

    with open(os.path.join("analysis_files", "environments", "{}.json".format(env)), "r+") as env_file:
        environment = json.load(env_file)

    if environment["basic_shape"] == "cuboid":
        for el in elements:
            with open(os.path.join("analysis_files", "segmentations", el), "r+") as seg_file:
                segmentations = json.load(seg_file)

            file_path = os.path.join("analysis_files", "segmentations", "{}_ENV_{}".format(el.split(".")[0], env))
            os.mkdir(file_path)

            for cls, val in segmentations.items():
                sampling.cuboid_segmentation(file_path, environment, cls, val, save=True)


def draw_samples(all_seg=True, all_dist=True, segmentation_sets=None, distribution_sets=None):
    """
    This function allows to draw samples from distributions based on given segmentations.
    :param all_seg: True if all available segmentation sets should be used
    :param all_dist: True if all available distribution sets should be used
    :param segmentation_sets: A list of segmentation sets that should be used, if all_seg is set False
    :param distribution_sets: A list of distribution sets that should be used, if all_dist is set False
    :return:
    """
    if all_seg:
        segmentation_sets = \
            [x for x in os.listdir(os.path.join("analysis_files", "segmentations")) if not x.endswith(".json")]
    if all_dist:
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


def run_training(sampling_on_the_fly, segmentation_sets=None, distribution_sets=None):
    if segmentation_sets is None:
        segmentation_sets = \
            [x for x in os.listdir(os.path.join("analysis_files", "segmentations")) if not x.endswith(".json")]
    if distribution_sets is None:
        distribution_sets = \
            [x for x in os.listdir(os.path.join("analysis_files", "distributions")) if not x.endswith(".json")]

    for segmentation_set in segmentation_sets:
        segmentation_set_results_path = os.path.join("analysis_files", "results", segmentation_set)
        os.mkdir(segmentation_set_results_path)
        segmentation_set_models_path = os.path.join("analysis_files", "models", segmentation_set)
        os.mkdir(segmentation_set_models_path)
        segmentations = os.listdir(os.path.join("analysis_files", "segmentations", segmentation_set))
        segmentations.sort()

        for distribution_set in distribution_sets:
            distribution_set_results_path = os.path.join(segmentation_set_results_path, distribution_set)
            os.mkdir(distribution_set_results_path)
            distribution_set_models_path = os.path.join(segmentation_set_models_path, distribution_set)
            os.mkdir(distribution_set_models_path)

            if sampling_on_the_fly:
                accuracies = {}
                for segmentation in segmentations:
                    segmentation_loaded = np.load(os.path.join("analysis_files", "segmentations",
                                                               segmentation_set, segmentation))
                    segmentation_results_path = os.path.join(distribution_set_results_path,
                                                             "segmentation_{}".format(segmentation.split(".")[0]))
                    os.mkdir(segmentation_results_path)
                    segmentation_models_path = os.path.join(distribution_set_models_path,
                                                            "segmentation_{}".format(segmentation.split(".")[0]))
                    os.mkdir(segmentation_models_path)
                    segmentation_name = segmentation.split(".")[0]
                    accuracy = training.train_with_sampling_on_the_fly(
                        segmentation_name,
                        segmentation_loaded,
                        segmentation_results_path,
                        segmentation_models_path,
                        distribution_set
                    )
                    accuracies[segmentation_name] = accuracy

                with open(os.path.join(distribution_set_results_path, "accuracies.json"), "w+") as acc_file:
                    pickle.dump(accuracies, acc_file)

            else:
                # TODO implement training with already drawn samples
                pass
