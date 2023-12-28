import unittest

from equation_discover import *


class TestSampler(unittest.TestCase):
    def setUp(self):
        self.sampler = RNNSampler(BASE_TOKENS, 16, 2)

    def test_instanciation(self):
        sampler = RNNSampler(BASE_TOKENS, 16, 2)
        self.assertIsNotNone(sampler)

    def test_non_nan_grad(self):
        log_probs_grad = []
        entropies_grad = []
        for _ in range(10):
            with tf.GradientTape(persistent=True) as tape:
                sequences, lengths, entropies, log_probs = self.sampler.sample(10)

            log_probs_grad.extend(
                [
                    tf.reduce_any(tf.math.is_nan(grad))
                    for grad in tape.gradient(log_probs, self.sampler.variables)
                ]
            )
            entropies_grad.extend(
                [
                    tf.reduce_any(tf.math.is_nan(grad))
                    for grad in tape.gradient(entropies, self.sampler.variables)
                ]
            )
        self.assertEqual(any(log_probs_grad), False, "Gradients are not defined")
        self.assertEqual(any(entropies_grad), False, "Gradients are not defined")

    def test_sample(self):
        n_samples = 32
        (sequences, lengths, entropies, log_probs) = self.sampler.sample(n_samples)
        self.assertEqual(sequences.shape[0], n_samples)
