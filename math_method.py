import numpy as np


def l1_normalize(x):
    return x / np.sum(np.abs(x))


def l2_normalize(x):
    return x / np.sqrt(np.sum(np.square(x)))


def gradient(f, **kwargs):
    para = list(kwargs.items())
    step = 1e-6
    minus_move = f(**{para[0][0]: para[0][1] - step})
    plus_move = f(**{para[0][0]: para[0][1] + step})
    grad = (plus_move - minus_move) / (2 * step)
    return grad


def batch_normalization(x, mean, variance, offset, scale, variance_epsilon):
    inv = np.sqrt(variance + variance_epsilon)
    if scale is not None:
        inv *= scale
    return x * inv.astype(x.dtype) + (offset - mean * inv if offset is not None else -mean * inv).astype(x.dtype)


def linear_combination(x, w, b):
    assert w.shape[-1] == x.shape[0] and w.shape[0] == b.shape[
        0], "given weights and bias matrix should be aligned for matrix computing"
    return w.dot(x) + b


if __name__ == '__main__':
    x = np.ones(100).reshape((25, 4)).T
    w = np.array([[2, 1, 3, 4], [2, 1, 8, 7], [2, 3, 4, 2]])
    b = np.array([[0.5], [1], [2]])
    # print(linear_combination(x, w, b))
    from functools import partial
    new_f = partial(linear_combination, x=x, b=b)
    print(gradient(new_f, w=w))
    # res = batch_normalization(x, 0, 1, 0.1, 0.02, 0.001)
    # print(res)
