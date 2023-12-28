from typing import Literal, TypedDict

import tensorflow as tf
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.models import Model

from .constants import TF_FLOAT_DTYPE, TF_INT_DTYPE
from .logger import getLogger
from .probabilities import (Constraint, MaxLengthConstraint,
                            MinLengthConstraint, MinVariableExpression,
                            non_zero_probs, normalize)
from .tf_utils import tf_append, tf_isin, tf_vstack
from .tokens import TokenSequence


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


def _sample_token():
    pass


def _get_parent_sibling():
    pass


def update_arity(
    tokens: tf.Tensor, current_arity: tf.Tensor, token_library: TokenSequence
):
    current_arity -= 1
    current_arity += (
        tf.cast(tf_isin(tokens, token_library.two_arity_tensor), dtype=TF_INT_DTYPE) * 2
    )
    current_arity += tf.cast(
        tf_isin(tokens, token_library.one_arity_tensor), dtype=TF_INT_DTYPE
    )
    return current_arity


def _get_next_input():
    pass


class RNNSamplerInput(TypedDict):
    input: tf.Tensor
    hidden: tf.Tensor


class RNNSampler(Sampler):
    def __init__(
        self,
        tokens: TokenSequence,
        hidden_size: int,
        num_layers: int = 1,
        type: Literal["rnn"] = "rnn",
        dropout: float = 0,
        min_length: int = 2,
        max_length: int = 15,
    ):
        super().__init__()
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

        self.logger = getLogger("RNN Sampler")

        self.input_tensor = tf.Variable(
            initial_value=tf.random.uniform(shape=(1, self.input_size)),
            trainable=True,
        )
        self.init_hidden = tf.Variable(
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
        input_tensor = tf.tile(self.input_tensor, (n, 1))
        hidden_tensor = tf.tile(self.init_hidden, (n, 1))
        self({"input": input_tensor, "hidden": hidden_tensor})

    def call(self, inputs: RNNSamplerInput, training=None, mask=None):
        output = inputs["input"]
        state = inputs["hidden"]
        for layer in self.recurrent_layers:
            output, state = layer(tf.expand_dims(output, axis=1), initial_state=state)

        output = self.projection_layer(output)
        return output, state

    def sample(self, n: int):
        sequences = tf.zeros((n, 0), dtype=TF_INT_DTYPE)
        mask = tf.zeros((n, 0), dtype=tf.bool)
        entropies = tf.zeros((n, 0), dtype=TF_FLOAT_DTYPE)
        log_probs = tf.zeros((n, 0), dtype=TF_FLOAT_DTYPE)

        input_tensor = tf.tile(self.input_tensor, (n, 1))
        hidden_tensor = tf.tile(self.init_hidden, (n, 1))

        # number of tokens that must be sampled to complete expression
        arities = tf.ones(n, dtype=TF_INT_DTYPE)
        still_alive = tf.ones(n, dtype=tf.bool)

        def body(
            input_tensor,
            hidden_tensor,
            arities,
            sequences,
            still_alive,
            mask,
            entropies,
            log_probs,
        ):
            tokens, log_prob, entropy = self._sample_tokens(
                input_tensor=input_tensor,
                hidden_tensor=hidden_tensor,
                counters=arities,
                sequences=sequences,
            )
            arities = update_arity(
                current_arity=arities, tokens=tokens, token_library=self.tokens
            )
            # is alive either because counters is still greater than 0 and is not dead yet.
            still_alive = tf.logical_and(arities > 0, still_alive)
            sequences = tf_append(sequences, tokens)

            parent_sibling = self._get_parent_sibling(sequences)
            input_tensor = self._get_next_input(parent_sibling)

            mask = tf_append(mask, still_alive)
            entropies = tf_append(entropies, entropy)
            log_probs = tf_append(log_probs, log_prob)

            return (
                input_tensor,
                hidden_tensor,
                arities,
                sequences,
                still_alive,
                mask,
                entropies,
                log_probs,
            )

        def cond(
            input_tensor,
            hidden_tensor,
            arities,
            sequences,
            still_alive,
            mask,
            entropies,
            log_probs,
        ):
            return tf.reduce_any(still_alive)

        (
            input_tensor,
            hidden_tensor,
            arities,
            sequences,
            still_alive,
            mask,
            entropies,
            log_probs,
        ) = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=[
                input_tensor,
                hidden_tensor,
                arities,
                sequences,
                still_alive,
                mask,
                entropies,
                log_probs,
            ],
            shape_invariants=[
                input_tensor.get_shape(),
                hidden_tensor.get_shape(),
                arities.get_shape(),
                tf.TensorShape((n, None)),
                still_alive.get_shape(),
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

    def _sample_tokens(self, input_tensor, hidden_tensor, counters, sequences):
        output, state = self({"input": input_tensor, "hidden": hidden_tensor})
        probs = tf.nn.softmax(output)
        probs = self.apply_constraints(
            output=probs, counters=counters, sequences=sequences
        )
        probs = non_zero_probs(probs)

        # compute probabilities
        self.logger.debug("Normalizing output", sum=tf.reduce_sum(probs, axis=1))
        token_probabilities = normalize(probs)
        token_log_probabilities = tf.math.log(token_probabilities)

        # sample categories and squeeze the resulting tensor
        tokens = tf.random.categorical(token_log_probabilities, 1, dtype=TF_INT_DTYPE)[
            :, 0
        ]
        indices = tf.concat(
            (tf.range(0, tokens.shape[0])[:, None], tokens[:, None]), axis=1
        )
        # get the log prob and entropies of sampled categories
        log_prob = tf.gather_nd(token_log_probabilities, indices=indices)
        # token_log_probabilities might be infinite if token_probabilities equals zeros
        entropy = tf.reduce_sum(
            tf.where(
                ~tf.math.is_inf(token_log_probabilities),
                -token_probabilities * token_log_probabilities,
                0,
            ),
            axis=1,
        )
        return tokens, log_prob, entropy

    def apply_constraints(
        self,
        output: tf.Tensor,
        counters: tf.Tensor,
        sequences: tf.Tensor,
    ):
        for constraint in self.constraints:
            output = constraint(
                probs=output,
                counters=counters,
                sequences=sequences,
            )

        return output

    def _get_parent_sibling(self, sequences):
        # parent_sibling set init to -1
        n_samples = sequences.shape[0]
        lengths = sequences.shape[1]
        parent_sibling = tf.ones((n_samples, 2), dtype=TF_INT_DTYPE) * -1
        c = tf.zeros(n_samples, dtype=TF_INT_DTYPE)

        recent = lengths - 1
        for i in range(recent, -1, -1):
            # determine arity of i-th tokens
            token_i = sequences[:, i]
            c = update_arity(tokens=token_i, current_arity=c, token_library=self.tokens)

            # In locations where c is zero (and parent_sibling is -1)
            # parent_sibling is set to sequences[i] and [i+1]
            # when on the last item of the sequence, pad with -1 to get (n, 2) tensor
            c_mask = tf.logical_and(
                c == 0, tf.reduce_all(parent_sibling == -1, axis=1)
            )[:, None]
            i_ip1 = sequences[:, i : i + 2]
            if i == recent:
                i_ip1 = tf.pad(i_ip1, tf.constant([[0, 0], [0, 1]]), constant_values=-1)
            # wet i_ip1 to 0 when c_mask is False
            i_ip1 = i_ip1 * tf.cast(c_mask, dtype=TF_INT_DTYPE)
            parent_sibling = parent_sibling * tf.cast(~c_mask, dtype=TF_INT_DTYPE)
            parent_sibling = parent_sibling + i_ip1

        # True for most recent token is non-zero arity False otherwise
        recent_non_zero_mask = ~tf_isin(
            sequences[:, recent], self.tokens.zero_arity_tensor
        )[:, None]
        parent_sibling = parent_sibling * tf.cast(
            ~recent_non_zero_mask, dtype=TF_INT_DTYPE
        )

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

    def _get_next_input(self, parent_sibling):
        parent, sibling = parent_sibling[:, 0], parent_sibling[:, 1]
        return tf.concat(
            [
                tf.one_hot(parent, depth=len(self.tokens)),
                tf.one_hot(sibling, depth=len(self.tokens)),
            ],
            axis=1,
        )
