import numpy as np
from analysis import functions
from scipy.stats import multivariate_normal


#######################################
# STANDARD-DISTRIBUTIONS ##############
#######################################

def draw_sample(params, seg, size, save=False, path=None):
    """
    Drawing a sample of a distribution through a segmentation and creating a grid with those values
    :param params: Distribution parameters used to create a multivariate-normal distribution
    :param seg: Segmentation, which values are used to draw values from distribution
    :param size: Length, width and height of the segmentation grid
    :param save: True if the sample should be saved
    :param path: If save is True, this should be the path to where the distributions will be saved
    :return: Array representing the grid filled with distribution values
    """
    dist = multivariate_normal(params[:3], np.diag(params[3:]))
    sample = dist.pdf(seg[:, 3:])
    grid = functions.turn_coordinates_into_grid(seg[:, :3], sample, size)
    if save:
        if path is not None:
            np.save(path, grid)
        else:
            print("Sample not saved, please provide a path")
    return grid


#######################################
# CLUSTERS-DISTRIBUTIONS ##############
#######################################

def create_multiple_distributions():
    # TODO: Implement creation of multiple distributions in one environment
    pass


#######################################
# TIME-SERIES-DISTRIBUTIONS ###########
#######################################

def create_time_series_distributions():
    # TODO: Implement creation of distributions for time-series-analysis
    # Different parts can be observed: change of position, change of size, relation between position and size
    pass
