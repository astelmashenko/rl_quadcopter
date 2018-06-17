import numpy as np


def udacity_distance(x, y):
    return (abs(x - y)).sum()


def eucl_distance(x, y):
    return np.linalg.norm(x - y)
