import os
import random

import numpy as np


#######################################
# SEGMENTATIONS #######################
#######################################

def create_cuboid_segmentation(env, param, save=False, file_path=None, segmentation_class=None):
    """
    Creation of a segmentation as a cuboid grid, of a cuboid environment
    :param env: The environment, which will be segmented; the env file needs to follow the same pattern as the example
    :param param: A list of values by which the environment will be divided in x, y and z dimension
    :param save: True if the segmentation should be saved
    :param file_path: If save is True, this should be the path to the directory where the segmentation will be saved
    :param segmentation_class: If save is True, this should be the name of the segmentation class
    :return: Array of the center points of all segments
    """
    centers = []
    length_part, width_part, height_part = param
    length, width, height = env["params"]["length"], env["params"]["width"], env["params"]["height"]
    seg_length = length / length_part
    seg_width = width / width_part
    seg_height = height / height_part
    for l_p in range(length_part):
        for w_p in range(width_part):
            for h_p in range(height_part):
                centers.append([
                    l_p,
                    w_p,
                    h_p,
                    seg_length * (l_p + 0.5),
                    seg_width * (w_p + 0.5),
                    seg_height * (h_p + 0.5)
                ])
    centers = np.array(centers)
    if save:
        if file_path is not None and segmentation_class is not None:
            np.save(os.path.join(file_path, segmentation_class), centers)
        else:
            print("Segmentations not saved, please provide path and class-name")
    return centers


#######################################
# STANDARD-DISTRIBUTIONS ##############
#######################################

def create_distributions(params, size, save=False, file_path=None, dist_class=None):
    """
    Creating distributions belonging to the same class
    :param params: The parameters describing in which range the
    :param size: The number of sample distributions to be created
    :param save: True if the distribution should be saved
    :param file_path: If save is True, this should be the path to the directory where the distributions will be saved
    :param dist_class: If save is True, this should be the name of the distribution class
    :return: Array containing all distribution samples
    """
    distributions = []
    for i in range(size):
        distributions.append([random.uniform(x[0], x[1]) if x[0] < x[1] else x[0] for x in params])
    if save:
        if file_path is not None and dist_class is not None:
            np.save(os.path.join(file_path, str(dist_class)), distributions)
        else:
            print("Distributions not saved, please provide path and class-name")
    return distributions


