import numpy as np


class Zeros():

    def __call__(self, shape, dtype=None):
        return np.zeros_like(shape)


class Ones():

    def __call__(self, shape, dtype=None):
        return np.ones_like(shape)


class Constant():

    def __init__(self, value=0):
        self.value = value

    def __call__(self, shape, dtype=None):
        return np.ones_like(shape) * self.value


class RandomNormal():

    def __init__(self, mean=0., std=0.0005, seed=None):
        self.mean = mean
        self.std = std
        self.seed = seed

    def __call__(self, shape, dtype=None, **kwargs):
        np.random.seed(self.seed)
        return np.random.normal(self.mean, self.std, shape)


class RandomUniform():

    def __init__(self, minval=-0.05, maxval=0.05, seed=None):
        self.minval = minval
        self.maxval = maxval
        self.seed = seed

    def __call__(self, shape, dtype=None, **kwargs):
        np.random.seed(self.seed)
        return np.random.uniform(self.minval, self.maxval, shape)


class MSRA():

    def __init__(self, seed=None):
        self.seed = seed

    def __call__(self, shape, num_layers, dtype=None, **kwargs):
        np.random.seed(self.seed)
        return np.random.randn(shape[0], shape[1])/np.sqrt(shape[0] / num_layers)
