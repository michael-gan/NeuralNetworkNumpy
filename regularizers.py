import numpy as np


def dropout(x, drop_out_rate):
    keep_prob = 1 - drop_out_rate
    mask = np.random.rand(*x.shape) < keep_prob
    x = x * mask
    x = x / keep_prob
    return x


def l1_regularization(w, l1=0.01):
    return np.sum(np.abs(w)) * l1


def l2_regularization(w, l2=0.01):
    return np.sum(np.square(w)) * l2 / 2


def l1_l2_regularization(w, l1=0.01, l2=0.01):
    return l1_regularization(w, l1=l1) + l2_regularization(w, l2=l2)