import unittest

from equation_discover import *


class TestSampler(unittest.TestCase):
    def test_instanciation(self):
        sampler = EquationSampler(BASE_TOKENS, 16, 2)
        self.assertIsNotNone(sampler)

    def test_sample(self):
        n_samples = 32
        sampler = EquationSampler(BASE_TOKENS, 16, 2)
        (
            sequences,
            entropies,
            log_probs,
            counters,
            lengths,
            sequence_mask,
        ) = sampler.sample_sequence(n_samples)
        self.assertEqual(sequences.shape[0], n_samples)
