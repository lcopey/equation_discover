"""Module containing main constants of the library"""

import numpy as np
import tensorflow as tf

TF_FLOAT_DTYPE = tf.float32
TF_INT_DTYPE = tf.int32
NP_FLOAT_DTYPE = np.float32
EPS = np.finfo(np.float32).resolution
MAX_EXP = np.log(np.finfo(np.float32).max)
