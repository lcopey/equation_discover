import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from functools import wraps
from .sample import Uniform
from .utils import cast_to_list, get_numpy_array, rmse


class ProxyModel(Sequential):
    def __init__(self, n_layers: int | list[int], activations: str | list[str]):
        super().__init__()
        self.n_layers = cast_to_list(n_layers)
        self.activations = cast_to_list(activations, len(self.n_layers))

        for neuron, activation in zip(self.n_layers, self.activations):
            self.add(Dense(neuron, activation=activation))

        # last layer
        self.add(Dense(1))
        self.mse = MeanSquaredError()
        self.optimizer = Adam(learning_rate=1e-3)
        self.compile(loss=self.mse, optimizer=self.optimizer)

    @wraps(Sequential.fit)
    def fit(self, X: np.ndarray, y: np.ndarray, *args, **kwargs):
        stop_early = EarlyStopping(monitor='val_loss', patience=50, start_from_epoch=500)
        super().fit(X, y, *args, callbacks=[stop_early], **kwargs)

    @wraps(Sequential.predict)
    def predict(self, X, *args, **kwargs):
        predictions = super().predict(X, *args, **kwargs)
        if predictions.shape[-1] == 1:
            return predictions.flatten()
        return predictions

    def rmse(self, X: np.ndarray, y: np.ndarray):
        y_pred = self(X)
        y_pred = get_numpy_array(y_pred)
        return rmse(y, y_pred)

    def translational_error(self, X: np.ndarray, i: int, j: int):
        offset = np.empty_like(X)
        delta = X[:, i] - X[:, j]
        offset[:, i] = offset[:, j] = Uniform.from_array(delta).sample(delta.shape[0])

        for n in range(X.shape[-1]):
            if n != i and n != j:
                offset[:, n] = 0

        delta = self(X) - self(X + offset)
        delta = get_numpy_array(delta)

        return rmse(delta)
