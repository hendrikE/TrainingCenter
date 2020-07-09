import os
import pickle

import numpy as np

from analysis import _future_sampling, _future_training


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
                        _future_sampling.draw_sample(element, segmentation_loaded, size, save=True, path=element_path)


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
                    accuracy = _future_training.train_with_sampling_on_the_fly(
                        segmentation_loaded,
                        segmentation_path,
                        distribution_set
                    )
                    accuracies[segmentation_name] = accuracy

                with open(os.path.join(distribution_set_path, "accuracies"), "wb") as acc_file:
                    pickle.dump(accuracies, acc_file)

            else:
                # TODO implement training with already drawn samples
                pass
