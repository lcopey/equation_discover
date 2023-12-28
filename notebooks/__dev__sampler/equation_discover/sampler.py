from typing import TYPE_CHECKING, Literal, TypedDict

import tensorflow as tf
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.models import Model

from .constants import TF_FLOAT_DTYPE, TF_INT_DTYPE
from .logger import getLogger
from .tf_utils import tf_append, tf_isin, tf_vstack
from .tokens import TokenSequence


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
    def get_mask(
        self, sampler: "RNNSampler", counters: tf.Tensor, sequences: tf.Tensor
    ):
        lengths = sequences.shape[1]
        # mask (n_samples, 1)
        # 0 when a sequence has not yet reached sampler.min_lengths
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
        # min_length_mask = tf.cast(
        #     tf.logical_or(
        #         sampler.tokens.non_zero_arity_mask[None, :],
        #         (counters + lengths >= sampler.min_lengths)[:, None],
        #     ),
        #     dtype=TF_FLOAT_DTYPE,
        # )
        return min_length_mask

    def __call__(
        self,
        sampler: "RNNSampler",
        output: tf.Tensor,
        counters: tf.Tensor,
        sequences: tf.Tensor,
    ):
        min_length_mask = tf.stop_gradient(self.get_mask(sampler, counters, sequences))
        # zero out all terminal node
        output = tf.minimum(output, min_length_mask)

        return output


class MaxLengthConstraint(Constraint):
    # TODO init with max_length to be defined
    def __init__(self, max_length: int = 15):
        self.max_length = max_length
        self.logger = getLogger("MaxLengthConstraint")

    def get_mask(
        self, sampler: "RNNSampler", counters: tf.Tensor, sequences: tf.Tensor
    ):
        lengths = sequences.shape[1]
        max_length_mask = tf.cast(
            counters + lengths
            <= tf.ones(counters.shape, dtype=TF_INT_DTYPE) * (self.max_length - 2),
            dtype=TF_FLOAT_DTYPE,
        )[:, None]
        max_length_mask = tf.maximum(
            tf.cast(sampler.tokens.zero_arity_mask, dtype=TF_FLOAT_DTYPE)[None, :],
            max_length_mask,
        )
        return max_length_mask

    def __call__(
        self,
        sampler: "RNNSampler",
        output: tf.Tensor,
        counters: tf.Tensor,
        sequences: tf.Tensor,
    ):
        max_length_mask = self.get_mask(sampler, counters, sequences)
        self.logger.debug("Applying max length constraint", mask=max_length_mask)
        # max_length_mask = tf.stop_gradient(max_length_mask)
        output = tf.minimum(output, max_length_mask)

        return output


class MinVariableExpression(Constraint):
    def get_mask(
        self, sampler: "RNNSampler", counters: tf.Tensor, sequences: tf.Tensor
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
            return nonvar_zeroarity_mask

    def __call__(
        self,
        sampler: "RNNSampler",
        output: tf.Tensor,
        counters: tf.Tensor,
        sequences: tf.Tensor,
    ):
        nonvar_zeroarity_mask = tf.stop_gradient(
            self.get_mask(sampler, counters, sequences)
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


class RNNSamplerInput(TypedDict):
    input: tf.Tensor
    state: tf.Tensor


class RNNSampler(Sampler):
    def __init__(
        self,
        tokens: TokenSequence,
        hidden_size: int,
        num_layers: int = 1,
        type: Literal["rnn"] = "rnn",
        dropout: float = 0,
        min_lengths: int = 2,
    ):
        super().__init__()
        self.input_size = 2 * len(tokens)
        self.output_size = len(tokens)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.tokens = tokens
        self.type = type
        self.min_lengths = min_lengths
        self.constraints = [
            # MinLengthConstraint(),
            MaxLengthConstraint(),
            # MinVariableExpression(),
        ]

        self.logger = getLogger("RNN Sampler")

        self.inputs = tf.Variable(
            initial_value=tf.random.uniform(shape=(1, self.input_size)),
            trainable=True,
            name="input",
        )
        self.states = tf.Variable(
            initial_value=tf.random.uniform(shape=(1, self.hidden_size)),
            trainable=True,
            name="state",
        )

        if self.type == "rnn":
            self.recurrent_layers = [
                SimpleRNN(
                    units=self.hidden_size,
                    activation="tanh",
                    use_bias=True,
                    bias_initializer="zeros",
                    dropout=self.dropout,
                    return_sequences=False,
                    return_state=True,
                    stateful=False,
                )
                for _ in range(self.num_layers)
            ]
            self.projection_layer = Dense(
                units=self.output_size,
                bias_initializer="zeros",
            )

        # build model
        n = 32
        inputs = tf.tile(self.inputs, (n, 1))
        states = tf.tile(self.states, (n, 1))
        self({"input": inputs, "state": states})

    def call(self, inputs: RNNSamplerInput, training=None, mask=None):
        outputs = inputs["input"]
        states = inputs["state"]
        for layer in self.recurrent_layers:
            outputs, states = layer(
                tf.expand_dims(outputs, axis=1), initial_state=states
            )

        outputs = self.projection_layer(outputs)
        return outputs, states
