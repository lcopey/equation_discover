import unittest

from equation_discover import *


class TestSampler(unittest.TestCase):
    def test_instanciation(self):
        sampler = RNNSampler(BASE_TOKENS, 16, 2)
        self.assertIsNotNone(sampler)

    def test_sample(self):
        n_samples = 32
        sampler = RNNSampler(BASE_TOKENS, 16, 2)
        (sequences, lengths, entropies, log_probs) = sampler.sample(n_samples)
        self.assertEqual(sequences.shape[0], n_samples)
