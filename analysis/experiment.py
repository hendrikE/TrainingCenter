# TODO: create an experimental setup that samples data with different parameters, train on it and compare results
import os
import json

import numpy as np

from analysis import sampling


def showcase_example():
    pass


def create_class_distributions():
    with open(os.path.join("analysis_files", "distributions", "classes.json"), "r+") as class_file:
        classes = json.load(class_file)

    for cls, val in classes.items():
        sampling.create_distributions(cls, np.array(val), size=500, save=True)


def create_time_series_distributions():
    pass


def create_segmentations(env):
    with open(os.path.join("analysis_files", "environments", "{}.json".format(env)), "r+") as env_file:
        environment = json.load(env_file)

    if environment["basic_shape"] == "cuboid":

        with open(os.path.join("analysis_files", "segmentations", "cuboid.json"), "r+") as seg_file:
            segmentations = json.load(seg_file)

        for cls, val in segmentations.items():
            sampling.cuboid_segmentation(env, environment, cls, val, save=True)


def draw_samples():
    segmentations = os.listdir(os.path.join("analysis_files", "segmentations"))
    distributions = os.listdir(os.path.join("analysis_files", "distributions"))
    for seg in segmentations:
        if seg.endswith(".npy"):
            seg_loaded = np.load(os.path.join("analysis_files", "segmentations", seg))
            name = seg.split(".")[0]
            path = os.path.join("analysis_files", "samples", name)
            os.mkdir(path)
            for dist in distributions:
                if dist.endswith(".npy"):
                    name = dist.split(".")[0]
                    path_cls = os.path.join(path, name)
                    os.mkdir(path_cls)
                    objects = np.load(os.path.join("analysis_files", "distributions", dist))
                    for index, obj in enumerate(objects):
                        sampling.draw_sample(obj, seg_loaded, path_cls, index)


def run_training():
    pass
