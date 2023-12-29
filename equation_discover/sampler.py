from typing import Literal, TypedDict

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.models import Model

from .constants import TF_FLOAT_DTYPE, TF_INT_DTYPE
from .logger import getLogger
from .probabilities import (Constraint, MaxLengthConstraint,
                            MinLengthConstraint, MinVariableExpression,
                            non_zero_probs, normalize)
from .tf_utils import tf_append, tf_isin, tf_vstack
from .tokens import TokenLibrary


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


def sample_token_distribution(
    sampler: "Sampler",
    inputs: tf.Tensor,
    state: tf.Tensor,
    arities: tf.Tensor,
    sequences: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    probs = sampler(
        {"input": inputs, "hidden": state, "arities": arities, "sequences": sequences}
    )
    dist = tfp.distributions.Categorical(probs=probs)
    tokens = dist.sample()
    return tokens, dist.log_prob(tokens), dist.entropy()


def update_arity(
    tokens: tf.Tensor, current_arity: tf.Tensor, token_library: TokenLibrary
):
    current_arity -= 1
    current_arity += (
        tf.cast(tf_isin(tokens, token_library.two_arity_tensor), dtype=TF_INT_DTYPE) * 2
    )
    current_arity += tf.cast(
        tf_isin(tokens, token_library.one_arity_tensor), dtype=TF_INT_DTYPE
    )
    return current_arity


def get_parent_sibling(token_library: TokenLibrary, sequences: tf.Tensor):
    # parent_sibling set init to -1
    n_samples = sequences.shape[0]
    lengths = sequences.shape[1]
    parent_sibling = tf.ones((n_samples, 2), dtype=TF_INT_DTYPE) * -1
    c = tf.zeros(n_samples, dtype=TF_INT_DTYPE)

    recent = lengths - 1
    for i in range(recent, -1, -1):
        # determine arity of i-th tokens
        token_i = sequences[:, i]
        c = update_arity(tokens=token_i, current_arity=c, token_library=token_library)

        # In locations where c is zero (and parent_sibling is -1)
        # parent_sibling is set to sequences[i] and [i+1]
        # when on the last item of the sequence, pad with -1 to get (n, 2) tensor
        c_mask = tf.logical_and(c == 0, tf.reduce_all(parent_sibling == -1, axis=1))[
            :, None
        ]
        i_ip1 = sequences[:, i : i + 2]
        if i == recent:
            i_ip1 = tf.pad(i_ip1, tf.constant([[0, 0], [0, 1]]), constant_values=-1)
        # wet i_ip1 to 0 when c_mask is False
        i_ip1 = i_ip1 * tf.cast(c_mask, dtype=TF_INT_DTYPE)
        parent_sibling = parent_sibling * tf.cast(~c_mask, dtype=TF_INT_DTYPE)
        parent_sibling = parent_sibling + i_ip1

    # True for most recent token is non-zero arity False otherwise
    recent_non_zero_mask = ~tf_isin(
        sequences[:, recent], token_library.zero_arity_tensor
    )[:, None]
    parent_sibling = parent_sibling * tf.cast(~recent_non_zero_mask, dtype=TF_INT_DTYPE)

    recent_parent_sibling = tf.concat(
        [
            sequences[:, recent, None],
            -1 * tf.ones((n_samples, 1), dtype=TF_INT_DTYPE),
        ],
        axis=1,
    )

    recent_parent_sibling = recent_parent_sibling * tf.cast(
        recent_non_zero_mask, dtype=TF_INT_DTYPE
    )

    parent_sibling = parent_sibling + recent_parent_sibling

    return parent_sibling


def get_next_input(token_library: TokenLibrary, parent_sibling: tf.Tensor):
    parent, sibling = parent_sibling[:, 0], parent_sibling[:, 1]
    return tf.concat(
        [
            tf.one_hot(parent, depth=len(token_library)),
            tf.one_hot(sibling, depth=len(token_library)),
        ],
        axis=1,
    )


class RNNSamplerInput(TypedDict):
    input: tf.Tensor
    hidden: tf.Tensor
    arities: tf.Tensor
    sequences: tf.Tensor


class RNNSampler(Sampler):
    def __init__(
        self,
        tokens: TokenLibrary,
        hidden_size: int,
        num_layers: int = 1,
        type: Literal["rnn"] = "rnn",
        dropout: float = 0,
        min_length: int = 2,
        max_length: int = 15,
    ):
        super().__init__()
        self.logger = getLogger(object="RNNSampler")

        self.input_size = 2 * len(tokens)
        self.output_size = len(tokens)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.tokens = tokens
        self.type = type
        self.constraints: list[Constraint] = [
            MinLengthConstraint(self, min_length=min_length),
            MaxLengthConstraint(self, max_length=max_length),
            MinVariableExpression(self),
        ]

        self.inputs = tf.Variable(
            initial_value=tf.random.uniform(shape=(1, self.input_size)),
            trainable=True,
        )
        self.initial_state = tf.Variable(
            initial_value=tf.random.uniform(shape=(1, self.hidden_size)), trainable=True
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
        init: RNNSamplerInput = {
            "input": tf.tile(self.inputs, (n, 1)),
            "hidden": tf.tile(self.initial_state, (n, 1)),
            "arities": tf.ones(n, dtype=TF_INT_DTYPE),
            "sequences": tf.zeros((n, 0), dtype=TF_INT_DTYPE),
        }
        self(init)

    def call(self, inputs: RNNSamplerInput, training=None, mask=None):
        output = inputs["input"]
        state = inputs["hidden"]
        arities = inputs["arities"]
        sequences = inputs["sequences"]
        logger = self.logger.bind(func="call")

        for layer in self.recurrent_layers:
            output, state = layer(tf.expand_dims(output, axis=1), initial_state=state)

        output = self.projection_layer(output)
        probs = tf.nn.softmax(output)
        logger.debug("Probs before constraints", probs=probs)

        for constraint in self.constraints:
            probs = constraint(probs=probs, arities=arities, sequences=sequences)
        logger.debug("After constraints", probs=probs)
        probs = non_zero_probs(probs)
        probs = normalize(probs)
        logger.debug("After normalization", probs=probs)
        return probs

    def sample(self, n: int):
        sequences = tf.zeros((n, 0), dtype=TF_INT_DTYPE)
        mask = tf.zeros((n, 0), dtype=tf.bool)
        entropies = tf.zeros((n, 0), dtype=TF_FLOAT_DTYPE)
        log_probs = tf.zeros((n, 0), dtype=TF_FLOAT_DTYPE)

        inputs = tf.tile(self.inputs, (n, 1))
        current_state = tf.tile(self.initial_state, (n, 1))

        current_arities = tf.ones(n, dtype=TF_INT_DTYPE)
        still_incomplete = tf.ones(n, dtype=tf.bool)

        def body(
            inputs,
            current_state,
            current_arities,
            sequences,
            still_incomplete,
            mask,
            entropies,
            log_probs,
        ):
            tokens, log_prob, entropy = sample_token_distribution(
                sampler=self,
                inputs=inputs,
                state=current_state,
                arities=current_arities,
                sequences=sequences,
            )

            # Append tokens to current sequence and update arities
            sequences = tf_append(sequences, tokens)
            current_arities = update_arity(
                current_arity=current_arities, tokens=tokens, token_library=self.tokens
            )
            # entropies and log_prob are stored for loss computation
            entropies = tf_append(entropies, entropy)
            log_probs = tf_append(log_probs, log_prob)

            # Sequences are complete when arities reaches 0
            still_incomplete = tf.logical_and(current_arities > 0, still_incomplete)
            mask = tf_append(mask, still_incomplete)

            parent_sibling = get_parent_sibling(
                token_library=self.tokens, sequences=sequences
            )
            inputs = get_next_input(
                token_library=self.tokens, parent_sibling=parent_sibling
            )

            return (
                inputs,
                current_state,
                current_arities,
                sequences,
                still_incomplete,
                mask,
                entropies,
                log_probs,
            )

        def cond(
            inputs,
            current_state,
            current_arities,
            sequences,
            still_incomplete,
            mask,
            entropies,
            log_probs,
        ):
            return tf.reduce_any(still_incomplete)

        (
            inputs,
            current_state,
            current_arities,
            sequences,
            still_incomplete,
            mask,
            entropies,
            log_probs,
        ) = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=[
                inputs,
                current_state,
                current_arities,
                sequences,
                still_incomplete,
                mask,
                entropies,
                log_probs,
            ],
            shape_invariants=[
                inputs.get_shape(),
                current_state.get_shape(),
                current_arities.get_shape(),
                tf.TensorShape((n, None)),
                still_incomplete.get_shape(),
                tf.TensorShape((n, None)),
                tf.TensorShape((n, None)),
                tf.TensorShape((n, None)),
            ],
        )

        lengths = tf.reduce_sum(tf.cast(mask, dtype=TF_INT_DTYPE), axis=-1) + 1
        mask_float = tf.cast(mask, dtype=TF_FLOAT_DTYPE)

        entropies = tf.reduce_sum(entropies * mask_float, axis=-1)
        log_probs = tf.reduce_sum(log_probs * mask_float, axis=-1)
        return sequences, lengths, entropies, log_probs
