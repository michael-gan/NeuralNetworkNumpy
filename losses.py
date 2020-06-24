from activations import activations
from math_method import *


class loss_function:
    epsilon = 1e-07

    def __init__(self):
        pass

    @staticmethod
    def __pre_check(x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        return x

    @classmethod
    def mean_squared_error(cls, y_true, y_pred):
        y_true, y_pred = cls.__pre_check(y_true), cls.__pre_check(y_pred)
        return np.mean(np.square(y_pred - y_true), axis=-1)

    @classmethod
    def mean_absolute_error(cls, y_true, y_pred):
        y_true, y_pred = cls.__pre_check(y_true), cls.__pre_check(y_pred)
        return np.mean(np.abs(y_pred - y_true), axis=-1)

    @classmethod
    def mean_absolute_percentage_error(cls, y_true, y_pred):
        y_true, y_pred = cls.__pre_check(y_true), cls.__pre_check(y_pred)
        diff = np.abs((y_true - y_pred) / np.clip(np.abs(y_true), cls.epsilon, None))
        return 100. * np.mean(diff, axis=-1)

    @classmethod
    def mean_squared_logarithmic_error(cls, y_true, y_pred):
        y_true, y_pred = cls.__pre_check(y_true), cls.__pre_check(y_pred)
        first_log = np.log(np.clip(y_pred, cls.epsilon, None) + 1.)
        second_log = np.log(np.clip(y_true, cls.epsilon, None) + 1.)
        return np.mean(np.square(first_log - second_log), axis=-1)

    @classmethod
    def squared_hinge(cls, y_true, y_pred):
        y_true, y_pred = cls.__pre_check(y_true), cls.__pre_check(y_pred)
        return np.mean(np.square(np.maximum(1. - y_true * y_pred, 0.)), axis=-1)

    @classmethod
    def hinge(cls, y_true, y_pred):
        y_true, y_pred = cls.__pre_check(y_true), cls.__pre_check(y_pred)
        return np.mean(np.maximum(1. - y_true * y_pred, 0.), axis=-1)

    @classmethod
    def categorical_hinge(cls, y_true, y_pred):
        y_true, y_pred = cls.__pre_check(y_true), cls.__pre_check(y_pred)
        pos = np.sum(y_true * y_pred, axis=-1)
        neg = np.max((1. - y_true) * y_pred, axis=-1)
        return np.maximum(0., neg - pos + 1.)

    @classmethod
    def logcosh(cls, y_true, y_pred):
        y_true, y_pred = cls.__pre_check(y_true), cls.__pre_check(y_pred)

        def _logcosh(x):
            return x + activations.softplus(-2. * x) - np.log(2.)

        return np.mean(_logcosh(y_pred - y_true), axis=-1)

    @classmethod
    def kullback_leibler_divergence(cls, y_true, y_pred):
        y_true, y_pred = cls.__pre_check(y_true), cls.__pre_check(y_pred)
        y_true = np.clip(y_true, cls.epsilon, 1)
        y_pred = np.clip(y_pred, cls.epsilon, 1)
        return np.sum(y_true * np.log(y_true / y_pred), axis=-1)

    @classmethod
    def poisson(cls, y_true, y_pred):
        y_true, y_pred = cls.__pre_check(y_true), cls.__pre_check(y_pred)
        return np.mean(y_pred - y_true * np.log(y_pred + cls.epsilon), axis=-1)

    @classmethod
    def cosine_proximity(cls, y_true, y_pred):
        y_true, y_pred = cls.__pre_check(y_true), cls.__pre_check(y_pred)
        return -np.sum(l2_normalize(y_true) * l2_normalize(y_pred), axis=-1)

    @classmethod
    def categorical_crossentropy(cls, y_true, y_pred):
        y_true, y_pred = cls.__pre_check(y_true), cls.__pre_check(y_pred)
        y_pred = y_pred / y_pred.sum(-1).repeat(y_pred.shape[-1]).reshape(y_pred.shape)
        y_true = np.clip(y_true, cls.epsilon, 1. - cls.epsilon)
        y_pred = np.clip(y_pred, cls.epsilon, 1. - cls.epsilon)
        return -np.sum(y_true * np.log(y_pred), axis=-1)

    @classmethod
    def sparse_categorical_crossentropy(cls, y_true, y_pred):
        y_true, y_pred = cls.__pre_check(y_true), cls.__pre_check(y_pred)
        y_pred = np.clip(y_pred, cls.epsilon, 1. - cls.epsilon)
        output = y_true.flatten().astype(int)
        target = np.zeros((len(output), y_pred.shape[-1]))
        target[range(len(output)), output] = 1
        return cls.categorical_crossentropy(target.reshape(*y_pred.shape), y_pred)

    @classmethod
    def binary_crossentropy(cls, y_true, y_pred):
        y_true, y_pred = cls.__pre_check(y_true), cls.__pre_check(y_pred)
        y_true = np.clip(y_true, cls.epsilon, 1. - cls.epsilon)
        y_pred = np.clip(y_pred, cls.epsilon, 1. - cls.epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred), axis=-1)