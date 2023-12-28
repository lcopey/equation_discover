from typing import TYPE_CHECKING

import tensorflow as tf

from .constants import EPS, TF_FLOAT_DTYPE, TF_INT_DTYPE
from .tf_utils import tf_isin

if TYPE_CHECKING:
    from .sampler import RNNSampler


def non_zero_probs(probs: tf.Tensor) -> tf.Tensor:
    """Replace 0 probs with eps to avoid NaN in gradients."""
    return tf.where(probs == 0.0, EPS, probs)


def normalize(probs: tf.Tensor) -> tf.Tensor:
    return probs / tf.reduce_sum(probs, axis=1)[:, None]


class Constraint:
    def __init__(self, sampler: "Sampler"):
        self.sampler = sampler

    def __call__(
        self,
        probs: tf.Tensor,
        counters: tf.Tensor,
        sequences: tf.Tensor,
    ):
        raise NotImplementedError


class MinLengthConstraint(Constraint):
    def __init__(self, sampler: "RNNSampler", min_length: int = 2):
        super().__init__(sampler)
        self.min_length = min_length

    def get_mask(self, counters: tf.Tensor, sequences: tf.Tensor):
        lengths = sequences.shape[1]
        # mask (n_samples, 1)
        # 0 when a sequence has not yet reached sampler.min_lengths
        # min_boolean_mask = tf.cast(
        #     counters + lengths
        #     >= tf.ones(counters.shape, dtype=TF_INT_DTYPE) * sampler.min_lengths,
        #     dtype=TF_FLOAT_DTYPE,
        # )[:, None]
        #
        # # mask (n, n_operators)
        # min_length_mask = tf.maximum(
        #     tf.cast(sampler.tokens.non_zero_arity_mask, dtype=TF_FLOAT_DTYPE)[None, :],
        #     min_boolean_mask,
        # )
        min_length_mask = tf.cast(
            tf.logical_or(
                self.sampler.tokens.non_zero_arity_mask[None, :],
                (counters + lengths >= self.min_length)[:, None],
            ),
            dtype=TF_FLOAT_DTYPE,
        )
        return min_length_mask

    def __call__(
        self,
        probs: tf.Tensor,
        counters: tf.Tensor,
        sequences: tf.Tensor,
    ):
        min_length_mask = tf.stop_gradient(
            self.get_mask(counters=counters, sequences=sequences)
        )
        # zero out all terminal node
        probs = tf.minimum(probs, min_length_mask)

        return probs


class MaxLengthConstraint(Constraint):
    def __init__(self, sampler: "RNNSampler", max_length: int = 15):
        super().__init__(sampler)
        self.max_length = max_length

    def get_mask(self, counters: tf.Tensor, sequences: tf.Tensor):
        lengths = sequences.shape[1]
        max_length_mask = tf.cast(
            counters + lengths
            <= tf.ones(counters.shape, dtype=TF_INT_DTYPE) * (self.max_length - 2),
            dtype=TF_FLOAT_DTYPE,
        )[:, None]
        max_length_mask = tf.maximum(
            tf.cast(self.sampler.tokens.zero_arity_mask, dtype=TF_FLOAT_DTYPE)[None, :],
            max_length_mask,
        )
        return max_length_mask

    def __call__(
        self,
        probs: tf.Tensor,
        counters: tf.Tensor,
        sequences: tf.Tensor,
    ):
        max_length_mask = self.get_mask(counters=counters, sequences=sequences)
        max_length_mask = tf.stop_gradient(max_length_mask)
        probs = tf.minimum(probs, max_length_mask)

        return probs


class MinVariableExpression(Constraint):
    def get_mask(self, counters: tf.Tensor, sequences: tf.Tensor):
        lengths = sequences.shape[1]
        # non zero arity or non variable
        nonvar_zeroarity_mask = tf.cast(
            ~tf.logical_and(
                self.sampler.tokens.zero_arity_mask,
                self.sampler.tokens.nonvariable_mask,
            ),
            dtype=TF_FLOAT_DTYPE,
        )

        if lengths == 0:
            return nonvar_zeroarity_mask

        else:
            nonvar_zeroarity_mask = tf.tile(
                nonvar_zeroarity_mask[None, :], multiples=(counters.shape[0], 1)
            )
            counter_mask = counters == 1

            if sequences.ndim == 1:
                sequences = sequences[:, None]
            contains_novar_mask = ~tf.reduce_any(
                tf_isin(
                    sequences,
                    self.sampler.tokens.variable_tensor
                    # tf.cast(sampler.tokens.variable_tensor, dtype=TF_DTYPE)
                ),
                axis=1,
            )

            last_token_and_no_var_mask = tf.cast(
                ~tf.logical_and(counter_mask, contains_novar_mask)[:, None],
                dtype=TF_FLOAT_DTYPE,
            )

            nonvar_zeroarity_mask = tf.maximum(
                nonvar_zeroarity_mask, last_token_and_no_var_mask
            )
            return nonvar_zeroarity_mask

    def __call__(
        self,
        probs: tf.Tensor,
        counters: tf.Tensor,
        sequences: tf.Tensor,
    ):
        nonvar_zeroarity_mask = tf.stop_gradient(self.get_mask(counters, sequences))
        probs = tf.minimum(probs, nonvar_zeroarity_mask)
        return probs
