import unittest

import numpy as np
import pandas as pd
import tensorflow as tf

from equation_discover import (BASE_TOKENS, RNNSampler, SymbolicLoss,
                               SymbolicRegressor, pandas_to_tensor)


class TestSymbolicRegressor(unittest.TestCase):
    def setUp(self):
        self.sampler = RNNSampler(BASE_TOKENS, 16, 1)
        self.regressor = SymbolicRegressor(
            self.sampler, n_samples=16, loss_func=SymbolicLoss()
        )
        X = pd.DataFrame(np.linspace(-2 * np.pi, 2 * np.pi), columns=["var_x"])
        y = np.sin((X * 2 + 1).squeeze())
        self.X = pandas_to_tensor(X)
        self.y = pandas_to_tensor(y)

    def test_regressor_outputs(self):
        outputs = self.regressor({"X": self.X, "y": self.y})
        self.assertTrue(
            tf.reduce_all(outputs.entropies > 0), "Entropies should be all positive"
        )
        self.assertTrue(
            tf.reduce_all(outputs.log_probs < 0), "Log probs should be all negative"
        )

    def test_regressor_grad(self):
        loss_func = SymbolicLoss()
        with tf.GradientTape() as tape:
            predictions = self.regressor({"X": self.X, "y": self.y})
            loss = loss_func(self.y, predictions)
        grads = tape.gradient(loss, self.regressor.sampler.variables)
        self.assertFalse(any([tf.reduce_any(tf.math.is_nan(grad)) for grad in grads]))
