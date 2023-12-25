from dataclasses import dataclass
from functools import wraps
from typing import Callable, Optional, TypedDict

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model

from .expressions import ExpressionEnsemble, pandas_to_tensor
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


class SymbolicLoss:
    def __init__(
        self,
        score_func: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
        risk_seeking: float = 0.1,
    ):
        self.risk_seeking = risk_seeking
        self.score_func = score_func

    def __call__(self, y_true, predictions: SymbolicRegressorOutput):
        rewards = self.score_func(y_true, predictions.y)
        threshold = np.quantile(rewards, 1 - self.risk_seeking)
        mask = rewards > threshold

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
        return risk_seeking_loss + entropy_loss


class SymbolicRegressor(Model):
    def __init__(
        self,
        sampler: Sampler,
        n_samples: int = 32,
        risk_seeking: float = 0.1,
    ):
        super().__init__()
        self.sampler = sampler
        self.n_samples = n_samples
        self.risk_seeking = risk_seeking
        self.build(tf.TensorShape(None))

    def call(
        self, inputs: SymbolicRegressorInputs, training=None, mask=None
    ) -> SymbolicRegressorOutput:
        X = inputs["X"]
        y = inputs["y"]
        (
            sequences,
            lengths,
            log_probs,
            entropies,
        ) = self.sampler.sample(self.n_samples)
        ensemble = ExpressionEnsemble(sequences, lengths)
        ensemble.optimize_constants(X=X, y=y)

        return SymbolicRegressorOutput(
            expressions=ensemble,
            y=ensemble.eval(X),
            # rewards=ensemble.score(X, y),
            entropies=entropies,
            log_probs=log_probs,
        )

    @wraps(Model.fit)
    def fit(self, x=None, y=None, **kwargs):
        x = pandas_to_tensor(x)
        y = pandas_to_tensor(y)
        inputs = {"X": x, "y": y}
        super().fit(x=inputs, y=y, **kwargs)
