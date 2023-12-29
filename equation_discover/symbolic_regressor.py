"""Module containing SymbolicRegressor."""
from dataclasses import dataclass
from typing import Callable, Optional, TypedDict

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model

from .constants import TF_FLOAT_DTYPE
from .expressions import ExpressionEnsemble, pandas_to_tensor
from .logger import TensorRepr, getLogger
from .rewards import rsquared
from .sampler import Sampler


class SymbolicRegressorInputs(TypedDict):
    X: pd.DataFrame | dict[str, tf.Tensor]
    y: pd.Series | tf.Tensor


@dataclass
class SymbolicRegressorOutput:
    expressions: ExpressionEnsemble
    y: tf.Tensor
    rewards: Optional[tf.Tensor] = None
    entropies: Optional[tf.Tensor] = None
    log_probs: Optional[tf.Tensor] = None

    # @property
    # def shape(self):
    #     return tf.TensorShape(len(self.expressions))
    #
    # @property
    # def dtype(self):
    #     return TF_FLOAT_DTYPE

    def __repr__(self):
        name_and_tensors = [
            (name, tensor)
            for name, tensor in zip(
                ("reward", "entropy", "log_prob"),
                (self.rewards, self.entropies, self.log_probs),
            )
            if tensor is not None
        ]
        names, tensors = zip(*name_and_tensors)
        inner_repr = "\n".join(
            [
                f"  {expr} {', '.join([f'{name}: {value:.1f}' for name, value in zip(names, values)])}"
                for expr, *values in zip(self.expressions, *tensors)
            ]
        )
        return f"<{self.__class__.__name__}\n{inner_repr}\n>"


class SymbolicLoss:
    def __init__(
        self,
        score_func: Callable[[tf.Tensor, tf.Tensor], tf.Tensor] = None,
        risk_seeking: float = 0.1,
    ):
        self.risk_seeking = risk_seeking
        if score_func is None:
            score_func = rsquared
        self.score_func = score_func
        self.logger = getLogger(object=self.__class__.__name__)

    def __call__(self, y_true: tf.Tensor, predictions: SymbolicRegressorOutput):
        rewards = self.score_func(y_true, predictions.y)
        threshold = np.quantile(rewards, 1 - self.risk_seeking)
        mask = rewards > threshold

        self.logger.debug(
            "Debugging", rewards=str(rewards), threshold=str(threshold), mask=str(mask)
        )

        # select best rewards
        best_rewards = rewards[mask]
        best_entropies = predictions.entropies[mask]
        best_log_probs = predictions.log_probs[mask]
        risk_seeking_loss = tf.clip_by_value(
            tf.reduce_sum((best_rewards - threshold) * best_log_probs)
            / best_rewards.shape[0],
            -1e6,
            1e6,
        )
        entropy_loss = tf.clip_by_value(
            tf.reduce_sum(best_entropies) / best_rewards.shape[0], -1e6, 1e6
        )
        self.logger.debug(
            "Debugging",
            risk_seeking=str(risk_seeking_loss),
            entropy_loss=str(entropy_loss),
        )
        return risk_seeking_loss + entropy_loss


class SymbolicRegressor(Model):
    def __init__(
        self,
        sampler: Sampler,
        loss_func: Optional[SymbolicLoss] = None,
        n_samples: int = 32,
    ):
        super().__init__()
        self.sampler = sampler
        self.n_samples = n_samples
        if loss_func:
            loss_func = SymbolicLoss()
        self.loss_func = loss_func
        self.build(tf.TensorShape(None))

    def call(self, inputs: SymbolicRegressorInputs, training=None, mask=None):
        X = inputs["X"]
        # y = inputs["y"]
        (
            sequences,
            lengths,
            entropies,
            log_probs,
        ) = self.sampler.sample(self.n_samples)
        ensemble = ExpressionEnsemble(sequences, lengths)
        # ensemble.optimize_constants(X=X, y=y)

        return SymbolicRegressorOutput(
            expressions=ensemble,
            y=ensemble.eval(X),
            entropies=entropies,
            log_probs=log_probs,
        )

    def fit(self, x=None, y=None, **kwargs):
        x = pandas_to_tensor(x)
        y = pandas_to_tensor(y)
        inputs = {"X": x, "y": y}
        with tf.GradientTape() as tape:
            predictions = self(inputs)
            loss = self.loss_func(y, predictions)

        grads = tape.gradient(loss, self.sampler.trainable_variables)
        return grads
