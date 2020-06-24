from losses import *


def binary_accuracy(y_true, y_pred):
    return np.mean(np.equal(y_true, np.round(y_pred)), axis=-1)


def categorical_accuracy(y_true, y_pred):
    return np.equal(np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1)).astype(float)


mse = MSE = loss_function.mean_squared_error
mae = MAE = loss_function.mean_absolute_error
mape = MAPE = loss_function.mean_absolute_percentage_error
msle = MSLE = loss_function.mean_squared_logarithmic_error
cosine = loss_function.cosine_proximity