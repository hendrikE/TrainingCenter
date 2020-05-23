import os
import pickle

import numpy as np
import pandas as pd


def calc_feature_values(array, percentages):
    values = []
    max_val = array.max()
    for percentage in percentages:
        above = np.nonzero(array >= max_val * percentage)[0][0]
        try:
            below = np.nonzero(array < max_val * percentage)[0][-1]
            values.append(below + 1 + (((max_val * percentage) - array[below]) / (array[above] - array[below])))
        except IndexError:
            values.append(1)
    return values


def turn_coordinates_into_grid(coordinates, values, size):
    grid = np.empty((size[0], size[1], size[2]))
    grid[:, :, :] = np.NaN
    for index, s in enumerate(coordinates):
        l, w, h = s
        grid[int(l), int(w), int(h)] = values[index]
    return grid


def clean_up_incompleteness(grid, size):
    replacement = [
        [-1, 0, 0],
        [-1, 1, 0],
        [-1, 1, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1]
    ]
    empty_indices = np.argwhere(np.isnan(grid)).tolist()
    if empty_indices:
        for empty_index in empty_indices:
            empty_index_replacements = []
            for x in replacement:
                if empty_index[0] + x[0] in range(size[0]) and empty_index[1] + x[1] in range(size[1]) and \
                        empty_index[2] + x[2] in range(size[2]):
                    val = grid[empty_index[0] + x[0], empty_index[1] + x[1], empty_index[2] + x[2]]
                    if not np.isnan(val):
                        empty_index_replacements.append(val)
            if empty_index_replacements:
                grid[empty_index[0], empty_index[1], empty_index[2]] = np.mean(empty_index_replacements)
            else:
                grid[empty_index[0], empty_index[1], empty_index[2]] = 0
    return grid


def turn_grid_into_features(grid, size):
    features = []

    x_dist = np.sum(grid, axis=(1, 2))
    if size[0] > 1:
        x_cdf = np.array([np.sum(x_dist[:i + 1]) for i in range(x_dist.shape[0])])
        features.append(calc_feature_values(x_cdf, [0.1, 0.25, 0.5, 0.75, 0.9]))
    else:
        features.append([x_dist[0] for _ in range(5)])

    y_dist = np.sum(grid, axis=(0, 2))
    if size[1] > 1:
        y_cdf = np.array([np.sum(y_dist[:i + 1]) for i in range(y_dist.shape[0])])
        features.append(calc_feature_values(y_cdf, [0.1, 0.25, 0.5, 0.75, 0.9]))
    else:
        features.append([y_dist[0] for _ in range(5)])

    z_dist = np.sum(grid, axis=(0, 1))
    if size[2] > 1:
        z_cdf = np.array([np.sum(z_dist[:i + 1]) for i in range(z_dist.shape[0])])
        features.append(calc_feature_values(z_cdf, [0.1, 0.25, 0.5, 0.75, 0.9]))
    else:
        features.append([z_dist[0] for _ in range(5)])

    return np.array(features).flatten()


def convert_results_to_csv(result_dirs, name):
    columns = ["segmentation_set", "segmentation", "distribution_set", "training_split",
               "incompleteness", "deviation", "classifier", "accuracy"]
    data = []
    for result_dir in result_dirs:
        parts = result_dir.split("_")
        training_split = parts[1]
        segmentation_set = parts[3] + parts[4] + parts[5] + parts[6]
        incompleteness = parts[8]
        deviation = parts[10]
        result_dir_path = os.path.join("analysis_files", "feature_results", result_dir)
        distribution_sets = os.listdir(result_dir_path)
        for distribution_set in distribution_sets:
            with open(os.path.join(result_dir_path, distribution_set, "accuracies"), "rb") as accuracy_file:
                accuracies = pickle.load(accuracy_file)
            for segmentation, results in accuracies.items():
                for classifier, accuracy in results.items():
                    data.append(
                        [segmentation_set, segmentation, distribution_set, training_split, incompleteness,
                         deviation, classifier, accuracy]
                    )
    df = pd.DataFrame(columns=columns, data=data)
    print(df)
    df.to_csv(os.path.join("analysis_files", "feature_results", "results_{}.csv".format(name)))
