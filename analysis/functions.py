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
