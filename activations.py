import warnings
import numpy as np
warnings.filterwarnings("ignore")


class activations:

    def __init__(self):
        pass

    @staticmethod
    def __pre_check(x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        return x

    @classmethod
    def softmax(cls, x):
        x = cls.__pre_check(x)
        ndim = np.ndim(x)
        if ndim == 1:
            raise ValueError('Cannot apply softmax to a array that is 1D')
        elif ndim >= 2:
            x_sum = np.sum(np.exp(x), axis=ndim-1, keepdims=True)
            x_exp = np.exp(x)
            return x_exp / x_sum
        else:
            raise ValueError('Cannot apply softmax to a tensor that is 1D. '
                             'Received input: %s' % x)

    @classmethod
    def elu(cls, x, alpha=1.0):
        x = cls.__pre_check(x)
        x[x < 0] = alpha*(np.exp(x[x < 0])-1)
        return x

    @classmethod
    def selu(cls, x):
        x = cls.__pre_check(x)
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * cls.elu(x, alpha)

    @classmethod
    def softplus(cls, x):
        x = cls.__pre_check(x)
        return np.log(np.exp(x) + 1)

    @classmethod
    def softsign(cls, x):
        x = cls.__pre_check(x)
        return x/(abs(x) + 1)

    @classmethod
    def relu(cls, x, alpha=0., max_value=None, threshold=0.):
        x = cls.__pre_check(x)
        x[x <= threshold] = alpha * (x[x <= threshold] - threshold)
        if max_value is None:
            return x
        else:
            return np.minimum(x, max_value)

    @classmethod
    def tanh(cls, x):
        x = cls.__pre_check(x)
        return np.tanh(x)

    @classmethod
    def sigmoid(cls, x):
        x = cls.__pre_check(x)
        return 1/(1 + np.exp(-x))

    @classmethod
    def hard_sigmoid(cls, x):
        x = cls.__pre_check(x)
        x[(x >= -2.5) & (x <= 2.5)] = 0.2 * x[(x >= -2.5) & (x <= 2.5)] + 0.5
        x[x < -2.5] = 0
        x[x > 2.5] = 1
        return x

    @classmethod
    def exponential(cls, x):
        x = cls.__pre_check(x)
        return np.exp(x)

    @classmethod
    def linear(cls, x):
        x = cls.__pre_check(x)
        return x
