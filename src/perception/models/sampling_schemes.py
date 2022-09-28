import numpy as np


def uniform_and_extremes_sampling():
    return np.random.choice([0.0, 1.0, np.random.rand()])

def uniform_sampling():
    return np.random.rand()

def extremes_sampling():
    return np.random.choice([0.0, 1.0])

def extremes_and_middle_sampling():
    return np.random.choice([0.0, 0.5, 1.0])

sampling_registry = {}
sampling_registry['uniform_sampling'] = uniform_sampling
sampling_registry['extremes_sampling'] = extremes_sampling
sampling_registry['extremes_and_middle_sampling'] = extremes_and_middle_sampling
sampling_registry['uniform_and_extremes_sampling'] = uniform_and_extremes_sampling