from typing import Literal

import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN, StringLookup
from tensorflow.keras.models import Model

from .tf_utils import TensorExpress, tf_bitwise, tf_isin
from .tokens import TokenSequence


class Constraint:
    def __call__(
        self,
        ssampler: "EquationSampler",
        output: tf.Tensor,
        counters: tf.Tensor,
        lengths: tf.Tensor,
        sequences: tf.Tensor,
    ):
        raise NotImplementedError


class MinLengthConstraint(Constraint):
    def __call__(
        self,
        sampler: "EquationSampler",
        output: tf.Tensor,
        counters: tf.Tensor,
        lengths: tf.Tensor,
        sequences: tf.Tensor,
    ):
        # mask (n_samples, 1)
        min_boolean_mask = tf.cast(
            counters + lengths
            >= tf.ones(counters.shape, dtype=tf.int32) * sampler.min_lengths,
            dtype=tf.float32,
        )[:, None]

        # mask (n, n_operators)
        min_length_mask = tf.maximum(
            tf.cast(sampler.tokens.non_zero_arity_mask, dtype=tf.float32)[None, :],
            min_boolean_mask,
        )

        # zero out all terminal node
        output = tf.minimum(output, min_length_mask)

        return output


class MaxLengthConstraint(Constraint):
    def __call__(
        self,
        sampler: "EquationSampler",
        output: tf.Tensor,
        counters: tf.Tensor,
        lengths: tf.Tensor,
        sequences: tf.Tensor,
    ):
        max_length_mask = tf.cast(
            counters + lengths <= tf.ones(counters.shape, dtype=tf.int32) * (12 - 2),
            dtype=tf.float32,
        )[:, None]
        max_length_mask = tf.maximum(
            tf.cast(sampler.tokens.zero_arity_mask, dtype=tf.float32)[None, :],
            max_length_mask,
        )
        output = tf.minimum(output, max_length_mask)

        return output


class MinVariableExpression(Constraint):
    def __call__(
        self,
        sampler: "EquationSampler",
        output: tf.Tensor,
        counters: tf.Tensor,
        lengths: tf.Tensor,
        sequences: tf.Tensor,
    ):
        # non zero arity or non variable
        nonvar_zeroarity_mask = tf.cast(
            ~tf.logical_and(
                sampler.tokens.zero_arity_mask, sampler.tokens.nonvariable_mask
            ),
            dtype=tf.float32,
        )

        if lengths[0] == 0:
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
                    # tf.cast(sampler.tokens.variable_tensor, dtype=tf.float32)
                ),
                axis=1,
            )

            last_token_and_no_var_mask = tf.cast(
                ~tf.logical_and(counter_mask, contains_novar_mask)[:, None],
                dtype=tf.float32,
            )

            nonvar_zeroarity_mask = tf.maximum(
                nonvar_zeroarity_mask, last_token_and_no_var_mask
            )

            output = tf.minimum(output, nonvar_zeroarity_mask)
            return output


class EquationSampler(Model):
    def __init__(
        self,
        tokens: TokenSequence,
        hidden_size: int,
        num_layers: int = 1,
        type: Literal["rnn"] = "rnn",
        dropout: float = 0,
        embedding_dim: int = 0,
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
            MinLengthConstraint(),
            MaxLengthConstraint(),
            MinVariableExpression(),
        ]

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
                for n in range(self.num_layers)
            ]
            self.projection_layer = Dense(
                units=self.output_size,
                bias_initializer="zeros",
            )

    def call(self, input_tensor, hidden_tensor):
        output = input_tensor
        state = hidden_tensor
        for layer in self.recurrent_layers:
            output, state = layer(tf.expand_dims(output, axis=1), initial_state=state)

        output = self.projection_layer(output)
        return output, state

    def sample_sequence(self, n: int):
        shape = (n, 0)
        sequences = tf.zeros(shape, dtype=tf.int32)
        entropies = tf.zeros(shape, dtype=tf.float32)
        log_probs = tf.zeros(shape, dtype=tf.float32)

        sequence_mask = tf.ones((n, 1), dtype=tf.bool)
        input_tensor = tf.tile(self.input_tensor, (n, 1))
        hidden_tensor = tf.tile(self.init_hidden, (n, 1))

        # number of tokens that must be sampled to complete expression
        counters = tf.ones(n, dtype=tf.int32)
        # number of tokens currently in expressions
        # TODO is lengths useful ?
        lengths = tf.zeros(n, dtype=tf.int32)

        while tf.reduce_any(tf.reduce_all(sequence_mask, axis=1)):
            tokens, log_prob, entropy = self.sample_tokens(
                input_tensor, hidden_tensor, counters, lengths, sequences
            )
            counters -= 1
            counters += (
                tf.cast(tf_isin(tokens, self.tokens.two_arity_tensor), dtype=tf.int32)
                * 2
            )
            counters += tf.cast(
                tf_isin(tokens, self.tokens.one_arity_tensor), dtype=tf.int32
            )

            lengths += 1

            # concat in the sequences
            sequence_mask = tf.concat(
                [
                    sequence_mask,
                    tf_bitwise((counters > 0), tf.reduce_all(sequence_mask, axis=1))[
                        :, None
                    ],
                ],
                axis=1,
            )

            sequences = tf.concat([sequences, tokens[:, None]], axis=1)
            entropies = tf.concat([entropies, entropy[:, None]], axis=1)
            log_probs = tf.concat([log_probs, log_prob[:, None]], axis=1)

            parent_sibling = self.get_parent_sibling(sequences, lengths)
            input_tensor = self.get_next_input(parent_sibling)

        lengths = TensorExpress(sequence_mask).int().sum(axis=1)
        return sequences, entropies, log_probs, counters, lengths, sequence_mask

    def sample_tokens(self, input_tensor, hidden_tensor, counters, lengths, sequences):
        output, state = self(input_tensor, hidden_tensor)
        output = tf.nn.softmax(output)
        output = self.apply_constraints(
            output=output, counters=counters, lengths=lengths, sequences=sequences
        )

        # compute probabilities
        token_probabilities = output / tf.reduce_sum(output, axis=1)[:, None]
        token_log_probabilities = tf.math.log(token_probabilities)

        # sample categories and squeeze the resulting tensor
        tokens = tf.random.categorical(token_log_probabilities, 1, dtype=tf.int32)[:, 0]
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
        lengths: tf.Tensor,
        sequences: tf.Tensor,
    ):
        for constraint in self.constraints:
            output = constraint(
                self,
                output=output,
                counters=counters,
                lengths=lengths,
                sequences=sequences,
            )

        return output

    def get_parent_sibling(self, sequences, lengths):
        parent_sibling = tf.ones((lengths.shape[0], 2), dtype=tf.int32) * -1

        c = tf.zeros(lengths.shape[0])

        recent = lengths[0].numpy() - 1
        for i in range(recent, -1, -1):
            # determine arity of i-th tokens
            token_i = sequences[:, i]
            arity = tf.zeros(lengths.shape[0])
            arity += (
                tf.cast(
                    tf_isin(token_i, self.tokens.two_arity_tensor), dtype=tf.float32
                )
                * 2
            )
            arity += tf.cast(
                tf_isin(token_i, self.tokens.one_arity_tensor), dtype=tf.float32
            )

            # Increment c by arity of the i-th toke minus 1
            c += arity
            c -= 1

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
            i_ip1 = i_ip1 * tf.cast(c_mask, dtype=tf.int32)
            parent_sibling = parent_sibling * tf.cast(~c_mask, dtype=tf.int32)
            parent_sibling = parent_sibling + i_ip1

        # True for most recent token is non-zero arity False otherwise
        recent_non_zero_mask = ~tf_isin(
            sequences[:, recent], self.tokens.zero_arity_tensor
        )[:, None]
        parent_sibling = parent_sibling * tf.cast(~recent_non_zero_mask, dtype=tf.int32)

        recent_parent_sibling = tf.concat(
            [
                sequences[:, recent, None],
                -1 * tf.ones((lengths.shape[0], 1), dtype=tf.int32),
            ],
            axis=1,
        )

        recent_parent_sibling = recent_parent_sibling * tf.cast(
            recent_non_zero_mask, dtype=tf.int32
        )

        parent_sibling = parent_sibling + recent_parent_sibling

        return parent_sibling

    def get_next_input(self, parent_sibling):
        parent, sibling = parent_sibling[:, 0], parent_sibling[:, 1]
        return tf.concat(
            [
                tf.one_hot(parent, depth=len(self.tokens)),
                tf.one_hot(sibling, depth=len(self.tokens)),
            ],
            axis=1,
        )
