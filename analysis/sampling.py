import os
import random

from scipy.stats import multivariate_normal
import numpy as np


def create_distribution(mx, my, mz, var_x, var_y, var_z):
    dist = multivariate_normal([mx, my, mz], [[var_x, 0, 0], [0, var_y, 0], [0, 0, var_z]])
    return dist


def create_distributions(file_path, dist_class, params, size, save=False):
    # dist = multivariate_normal(params[:, 0], np.diag(params[:, 1]))
    # distributions = dist.rvs(size=size, random_state=1)
    # distributions = np.array(distributions)
    # max_element = np.amax(distributions[:, 3:])
    # distributions[:, 3:] = np.where(distributions[:, 3:] <= 0,
    #                                 random.uniform(0, max_element),
    #                                 distributions[:, 3:])
    distributions = []
    for i in range(size):
        distributions.append([random.uniform(x[0], x[1]) if x[0] < x[1] else x[0] for x in params])
    if save:
        np.save(os.path.join(file_path, str(dist_class)), distributions)
    return distributions


def cuboid_segmentation(file_path, env, cls, param, save=False):
    centers = []
    length_part, width_part, height_part = param
    length, width, height = env["params"]["length"], env["params"]["width"], env["params"]["height"]
    seg_length = length / length_part
    seg_width = width / width_part
    seg_height = height / height_part
    for l in range(length_part):
        for w in range(width_part):
            for h in range(height_part):
                centers.append([
                    l,
                    w,
                    h,
                    seg_length * (l + 0.5),
                    seg_width * (w + 0.5),
                    seg_height * (h + 0.5)
                ])
    centers = np.array(centers)
    if save:
        np.save(os.path.join(file_path, str(cls)), centers)
    return centers


def cylinder_segmentation(env, param, save=False):
    centers = []
    circumference_part, height_part = param
    circumference, height = env["radius"], env["height"]
    segments = []
    for seg in segments:
        # TODO: implement formula for centroid of circular sector (https://en.wikipedia.org/wiki/List_of_centroids)
        centers.append(0)


def draw_sample(params, seg, path, size):
    dist = multivariate_normal(params[:3], np.diag(params[3:]))
    grid = np.zeros((size[0], size[1], size[2]))
    sample = dist.pdf(seg[:, 3:])
    for index, s in enumerate(seg[:, :3]):
        l, w, h = s
        grid[int(l), int(w), int(h)] = sample[index]
    np.save(path, grid)
