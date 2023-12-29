"""Module containing losses used in conjunction with SymbolicLoss"""
import tensorflow as tf
from tensorflow.keras.metrics import MSE


def RMSE(y_true, y_pred):
    return tf.sqrt(MSE(y_true, y_pred))


def normalized_rmse(y_true, y_pred):
    """Root mean squared error normalized with the standard deviation of y_true."""
    return tf.math.reduce_std(y_true) * RMSE(y_true, y_pred)


def rsquared(y_true, y_pred):
    return 1 / (1 + normalized_rmse(y_true, y_pred))
