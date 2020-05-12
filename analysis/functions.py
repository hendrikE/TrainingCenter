import numpy as np


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
