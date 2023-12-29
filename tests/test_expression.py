import unittest

from equation_discover import *


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

    def test_expression_fit_lbfgs(self):
        for sequence in [self.sequence_wo_const, self.sequence]:
            tree = Node.from_sequence(sequence=sequence, tokens=BASE_TOKENS)
            expression = Expression(tree)
            expression.optimize_constants(self.X, self.y, mode="lbfgs")
            self.assertIsNotNone(expression.res_)

    def test_expression_fit_basinhopping(self):
        for sequence in [self.sequence_wo_const, self.sequence]:
            tree = Node.from_sequence(sequence=sequence, tokens=BASE_TOKENS)
            expression = Expression(tree)
            expression.optimize_constants(self.X, self.y, mode="basinhopping")
            self.assertIsNotNone(expression.res_)


class TestExpressionEnsemble(unittest.TestCase):
    def setUp(self):
        n_samples = 32
        sampler = RNNSampler(BASE_TOKENS, 16, 2)
        (sequences, lengths, entropies, log_probs) = sampler.sample(n_samples)
        self.sequences = sequences
        self.lengths = lengths

    def test_build_tree(self):
        ensemble = ExpressionEnsemble(self.sequences, self.lengths)
        self.assertIsNotNone(ensemble)

    def test_eval(self):
        ensemble = ExpressionEnsemble(self.sequences, self.lengths)
        X = pd.DataFrame(np.linspace(-2 * np.pi, 2 * np.pi), columns=["var_x"])
        results = ensemble.eval(X)
        self.assertEqual(results.shape, (self.sequences.shape[0], X.shape[0]))
