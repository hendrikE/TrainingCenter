import random

import numpy as np
from analysis import functions
from scipy.stats import multivariate_normal


def generate_features(distributions, segmentation, size, save=False, path=None):
    """
    Generate Features based on a sample of a distribution through a segmentation
    :param distributions: Parameters of all distributions belonging to one class
    :param segmentation: Segmentation, which values are used to draw values from distributions
    :param size: Length, width and height of the segmentation grid
    :param save: True if the sample should be saved
    :param path: If save is True, this should be the path to where the features will be saved
    :return: Array containing features for all distributions
    """
    distributions_features = []
    for distribution in distributions:
        distribution = multivariate_normal(distribution[:3], np.diag(distribution[3:]))
        sample = distribution.pdf(segmentation[:, 3:])
        if type(sample) != np.ndarray:
            sample = [sample]

        grid = functions.turn_coordinates_into_grid(segmentation[:, :3], sample, size)
        distributions_features.append(functions.turn_grid_into_features(grid, size))
    if save:
        if path is not None:
            np.save(path, distributions_features)
        else:
            print("Features not saved, please provide a path")
    return distributions_features


def generate_features_with_deviation(distributions, segmentation_base, size, deviation, save=False, path=None):
    """
    Generate Features based on a sample of a distribution through a segmentation, where the segmentation points deviate
    :param distributions: Parameters of all distributions belonging to one class
    :param segmentation_base: Segmentation, which values are used to draw values from distributions
    :param size: Length, width and height of the segmentation grid
    :param deviation: Variance by how much each segmentation point will vary
    :param save: True if the sample should be saved
    :param path: If save is True, this should be the path to where the features will be saved
    :return: Array containing features for all distributions
    """
    distributions_features = []
    if deviation > 0.0:
        deviation_distributions = []
        for index in range(segmentation_base.shape[0]):
            deviation_distributions.append(multivariate_normal(segmentation_base[index, 3:], np.diag([deviation] * 3)))
        for distribution in distributions:
            segmentation = []
            for index in range(segmentation_base.shape[0]):
                segmentation.append(np.concatenate(segmentation_base[index, :3], deviation_distributions[index].rvs()))
            segmentation = np.array(segmentation)
            distribution = multivariate_normal(distribution[:3], np.diag(distribution[3:]))
            sample = distribution.pdf(segmentation[:, 3:])
            if type(sample) != np.ndarray:
                sample = [sample]

            grid = functions.turn_coordinates_into_grid(segmentation[:, :3], sample, size)
            distributions_features.append(functions.turn_grid_into_features(grid, size))
    else:
        for distribution in distributions:
            distribution = multivariate_normal(distribution[:3], np.diag(distribution[3:]))
            sample = distribution.pdf(segmentation_base[:, 3:])
            if type(sample) != np.ndarray:
                sample = [sample]

            grid = functions.turn_coordinates_into_grid(segmentation_base[:, :3], sample, size)
            distributions_features.append(functions.turn_grid_into_features(grid, size))

    if save:
        if path is not None:
            np.save(path, distributions_features)
        else:
            print("Features not saved, please provide a path")
    return distributions_features


def generate_features_with_incompleteness(distributions, segmentation, size, inc_prob, save=False, path=None):
    """
    Generate Features based on a sample of a distribution through a segmentation, with missing values of the sample
    :param distributions: Parameters of all distributions belonging to one class
    :param segmentation: Segmentation, which is used to draw values from distributions
    :param size: Length, width and height of the segmentation grid
    :param inc_prob: Probability that a value is missing
    :param save: True if the sample should be saved
    :param path: If save is True, this should be the path to where the features will be saved
    :return: Array containing features for all distributions
    """
    distributions_features = []
    for distribution in distributions:
        distribution = multivariate_normal(distribution[:3], np.diag(distribution[3:]))
        sample = distribution.pdf(segmentation[:, 3:])
        if type(sample) != np.ndarray:
            sample = np.array([sample])

        delete_indices = [index for index in range(segmentation.shape[0]) if random.random() < inc_prob]
        sample[delete_indices] = np.NaN

        grid = functions.turn_coordinates_into_grid(segmentation[:, :3], sample, size)
        grid = functions.clean_up_incompleteness(grid, size)
        distributions_features.append(functions.turn_grid_into_features(grid, size))

    if save:
        if path is not None:
            np.save(path, distributions_features)
        else:
            print("Features not saved, please provide a path")
    return distributions_features
