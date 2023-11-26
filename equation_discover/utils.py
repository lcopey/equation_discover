from typing import Any
import tensorflow as tf

import numpy as np


def cast_to_list(value: Any | list[Any], n: int = 1):
    if not isinstance(value, list):
        return [value] * n
    return value


def get_numpy_array(tensor: np.ndarray | tf.Tensor):
    if isinstance(tensor, tf.Tensor):
        return tensor.numpy()

    return tensor


def rmse(y_true: np.ndarray, y_pred: np.ndarray | None = None):
    n = y_true.shape[0]
    if y_pred is not None:
        return np.sqrt(np.sum((y_true - y_pred.flatten()) ** 2) / n)
    else:
        return np.sqrt(np.sum(y_true ** 2) / n)
