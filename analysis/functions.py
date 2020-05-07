import numpy as np


def calc_feature_values(array, percentages):
    values = []
    max_val = array.max()
    for percentage in percentages:
        above = np.nonzero(array >= max_val * percentage)[0][0]
        below = np.nonzero(array < max_val * percentage)[0][-1]
        print(below)
        print(np.nonzero(array < max_val * percentage))
        values.append(below + 1 + (((max_val * percentage) - array[below]) / (array[above] - array[below])))
    return values
