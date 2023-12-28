import tensorflow as tf
import tensorflow_probability as tfp

from .constants import TF_FLOAT_DTYPE, TF_INT_DTYPE
from .logger import getLogger
from .sampler import RNNSampler
from .tf_utils import tf_append, tf_isin
from .tokens import BASE_TOKENS, TokenSequence

logger = getLogger("dev")


# def max_length_constraint(
#         sampler: "RNNSampler",
#         arities: tf.Tensor,
#         sequences: tf.Tensor,
#         still_alive: tf.Tensor,
#         current_iteration: float
# ):
#     max_length = 6.
#     # lengths = sequences.shape[1]
#
#     max_length_mask = tf.cast(
#         arities + current_iteration <= max_length - 2,
#         dtype=TF_FLOAT_DTYPE,
#     )[:, None]
#     max_length_mask = tf.maximum(
#         tf.cast(sampler.tokens.zero_arity_mask, dtype=TF_FLOAT_DTYPE)[None, :],
#         max_length_mask,
#     )
#     return max_length_mask


def update_arity(
    tokens: tf.Tensor, current_arity: tf.Tensor, token_library: TokenSequence
):
    current_arity -= 1.0
    current_arity += (
        tf.cast(tf_isin(tokens, token_library.two_arity_tensor), dtype=TF_FLOAT_DTYPE)
        * 2.0
    )
    current_arity += tf.cast(
        tf_isin(tokens, token_library.one_arity_tensor), dtype=TF_FLOAT_DTYPE
    )
    return current_arity


def eval_model(sampler: RNNSampler, inputs, n_sample: int):
    # init
    arities = tf.ones(n_sample, dtype=TF_FLOAT_DTYPE)
    sequences = tf.zeros((n_sample, 0), dtype=TF_INT_DTYPE)
    still_alive = tf.ones(n_sample, dtype=tf.bool)
    mask = tf.zeros((n_sample, 0), dtype=tf.bool)
    log_probs = tf.zeros((n_sample, 0), dtype=TF_FLOAT_DTYPE)
    current_iteration = 10

    for _ in range(1):
        # sample tokens
        probs = get_token_probabilities(
            sampler=sampler,
            inputs=inputs,
            n_samples=n_sample,
            arities=arities,
            sequences=sequences,
            still_alive=still_alive,
            current_iteration=current_iteration,
        )

        # return probs, arities, sequences, still_alive, current_iteration
        # return probs
        dist = tfp.distributions.Categorical(probs=probs)
        tokens = dist.sample()
        logger.debug(tokens=tokens, return_line=True)
        log_prob = dist.log_prob(tokens)
        entropy = dist.entropy()

        arities = update_arity(tokens, arities, BASE_TOKENS)
        still_alive = tf.logical_and(arities > 0.0, still_alive)
        logger.debug(
            "Update arity", arities=arities, still_alive=still_alive, return_line=True
        )

        sequences = tf_append(sequences, tokens)
        mask = tf_append(mask, still_alive)
        log_probs = tf_append(log_probs, log_prob)
        logger.debug(sequences=sequences, mask=mask, return_line=True)
    return log_prob


def get_token_probabilities(
    sampler: RNNSampler,
    inputs: tf.Tensor,
    n_samples: int,
    arities: tf.Tensor,
    sequences: tf.Tensor,
    still_alive: tf.Tensor,
    current_iteration: float,
):
    outputs, _ = sampler(inputs)
    logger.debug(logits=outputs, return_line=True)

    probs = tf.nn.softmax(outputs)
    probs = tf.tile(probs, (n_samples, 1))
    logger.debug(raw_probabilities=probs, return_line=True)
    # return probs
    constraint = MaxLengthConstraint(sampler, max_length=4)
    probs = constraint(
        probs=probs,
        arities=arities,
        sequences=sequences,
        still_alive=still_alive,
        current_iteration=current_iteration,
    )

    return probs


class MaxLengthConstraint:
    def __init__(self, sampler: RNNSampler, max_length: int = 15):
        self.max_length = max_length
        self.sampler = sampler

    def mask(self, arities: tf.Tensor, sequences: tf.Tensor, current_iteration: float):
        max_length_mask = tf.cast(
            arities + current_iteration <= self.max_length - 2,
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
        arities: tf.Tensor,
        sequences: tf.Tensor,
        still_alive: tf.Tensor,
        current_iteration: float,
    ):
        mask = self.mask(
            arities=arities, sequences=sequences, current_iteration=current_iteration
        )
        if tf.reduce_any(mask != 1):
            logger.debug(
                "Applying max length constraint",
                max_length_constraint=mask,
                return_line=True,
            )
        else:
            logger.debug("No max length constraint")

        probs = tf.minimum(probs, mask)
        probs += 1e-20
        logger.debug(constrained_probs=probs, return_line=True)
        probs = probs / tf.reduce_sum(probs, axis=1)[:, None]
        logger.debug(normalized_probs=probs, return_line=True)
        return probs


# def apply_in_situ_constraint(outputs: tf.Tensor,
#                              sampler: RNNSampler,
#                              arities: tf.Tensor,
#                              sequences: tf.Tensor,
#                              still_alive: tf.Tensor,
#                              current_iteration: float):
#     max_length_constraint = MaxLengthConstraint(sampler, 4)
#     # mask = max_length_constraint(sampler=sampler,
#     #                              arities=arities,
#     #                              sequences=sequences,
#     #                              still_alive=still_alive,
#     #                              current_iteration=current_iteration)
#     mask = tf.convert_to_tensor([[m != outputs.shape[1] - 1 for m in range(outputs.shape[1])]
#                                  for n in range(outputs.shape[0])], dtype=TF_FLOAT_DTYPE)
#
#     if tf.reduce_any(mask != 1):
#         logger.debug(
#             "Applying max length constraint",
#             max_length_constraint=mask,
#             return_line=True,
#         )
#     else:
#         logger.debug("No max length constraint")
#     outputs = tf.minimum(outputs, mask)
#     logger.debug(constrained=outputs, return_line=True)
#
#     outputs = outputs / tf.reduce_sum(outputs, axis=1)[:, None]
#     logger.debug(normalized=outputs, return_line=True)
#     return outputs
