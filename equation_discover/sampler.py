import tensorflow as tf

from tensorflow.keras.models import Model

from .constants import TF_FLOAT_DTYPE, TF_INT_DTYPE
from .tf_utils import tf_isin


class Constraint:
    def __call__(
            self,
            ssampler: "RNNSampler",
            output: tf.Tensor,
            counters: tf.Tensor,
            sequences: tf.Tensor,
    ):
        raise NotImplementedError


class MinLengthConstraint(Constraint):
    def __call__(
            self,
            sampler: "RNNSampler",
            output: tf.Tensor,
            counters: tf.Tensor,
            sequences: tf.Tensor,
    ):
        lengths = sequences.shape[1]
        # mask (n_samples, 1)
        min_boolean_mask = tf.cast(
            counters + lengths
            >= tf.ones(counters.shape, dtype=TF_INT_DTYPE) * sampler.min_lengths,
            dtype=TF_FLOAT_DTYPE,
        )[:, None]

        # mask (n, n_operators)
        min_length_mask = tf.maximum(
            tf.cast(sampler.tokens.non_zero_arity_mask, dtype=TF_FLOAT_DTYPE)[None, :],
            min_boolean_mask,
        )

        # zero out all terminal node
        output = tf.minimum(output, min_length_mask)

        return output


class MaxLengthConstraint(Constraint):
    def __call__(
            self,
            sampler: "RNNSampler",
            output: tf.Tensor,
            counters: tf.Tensor,
            sequences: tf.Tensor,
    ):
        lengths = sequences.shape[1]
        max_length_mask = tf.cast(
            counters + lengths
            <= tf.ones(counters.shape, dtype=TF_INT_DTYPE) * (12 - 2),
            dtype=TF_FLOAT_DTYPE,
        )[:, None]
        max_length_mask = tf.maximum(
            tf.cast(sampler.tokens.zero_arity_mask, dtype=TF_FLOAT_DTYPE)[None, :],
            max_length_mask,
        )
        output = tf.minimum(output, max_length_mask)

        return output


class MinVariableExpression(Constraint):
    def __call__(
            self,
            sampler: "RNNSampler",
            output: tf.Tensor,
            counters: tf.Tensor,
            sequences: tf.Tensor,
    ):
        lengths = sequences.shape[1]
        # non zero arity or non variable
        nonvar_zeroarity_mask = tf.cast(
            ~tf.logical_and(
                sampler.tokens.zero_arity_mask, sampler.tokens.nonvariable_mask
            ),
            dtype=TF_FLOAT_DTYPE,
        )

        if lengths == 0:
            output = tf.minimum(output, nonvar_zeroarity_mask)
            return output

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
                    sampler.tokens.variable_tensor
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

            output = tf.minimum(output, nonvar_zeroarity_mask)
            return output


class Sampler(Model):
    def sample(self, n: int) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Returns sequences, lengths, entropies, log_probs"""
        raise NotImplementedError

    def apply_constraints(
            self,
            output: tf.Tensor,
            counters: tf.Tensor,
            sequences: tf.Tensor,
    ):
        raise NotImplementedError
