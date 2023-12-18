import unittest

from equation_discover import *


class TestExpression(unittest.TestCase):
    def setUp(self):
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
        self.sequences = sequences
        self.lengths = lengths

    def test_build_tree(self):
        for sequence, length in zip(self.sequences, self.lengths):
            tree = Node.build_tree(sequence[0:length], BASE_TOKENS)
            self.assertIsNotNone(tree)
