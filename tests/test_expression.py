import unittest

from equation_discover import *


class TestExpressionEnsemble(unittest.TestCase):
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
            tree = Node.from_sequence(sequence[0:length], BASE_TOKENS)
            self.assertIsNotNone(tree)


class TestExpression(unittest.TestCase):
    def setUp(self):
        self.sequence = [
            BASE_TOKENS.symbols.index(value)
            for value in ["sin", "+", "*", "const", "var_x", "const"]
        ]
        self.sequence_wo_const = [
            BASE_TOKENS.symbols.index(value) for value in ["sin", "var_x"]
        ]
        self.X = pd.DataFrame(np.linspace(-2 * np.pi, 2 * np.pi), columns=["var_x"])
        self.y = np.sin((self.X * 2 + 1).squeeze())

    def test_expression_eval(self):
        for sequence in [self.sequence_wo_const, self.sequence]:
            tree = Node.from_sequence(sequence=sequence, tokens=BASE_TOKENS)

            expression = Expression(tree)
            results = expression.eval(self.X)
            self.assertEqual(results.shape[0], self.X.shape[0])

    def test_expression_fit(self):
        for sequence in [self.sequence_wo_const, self.sequence]:
            tree = Node.from_sequence(sequence=sequence, tokens=BASE_TOKENS)
            expression = Expression(tree)
            expression.fit(self.X, self.y)
            self.assertIsNotNone(expression.res_)
