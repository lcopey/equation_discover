"""Protected function to evaluate expression"""
import tensorflow as tf

from .constants import MAX_EXP


def protected_log(x: tf.Tensor) -> tf.Tensor:
    x = tf.abs(x)
    return tf.where(
        x > 1e-3,
        tf.math.log(x),
        0.0,
    )


def protected_div(x1: tf.Tensor, x2: tf.Tensor) -> tf.Tensor:
    return tf.where(tf.abs(x2) > 1e-3, x1 / x2, 1.0)


def protected_exp(x1: tf.Tensor) -> tf.Tensor:
    return tf.where(x1 < MAX_EXP, tf.math.exp(x1), 0.0)
