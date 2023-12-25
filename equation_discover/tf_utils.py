import tensorflow as tf


def tf_isin(targets, values):
    return tf.reduce_any(tf.equal(tf.expand_dims(targets, axis=1), values), axis=1)


def tf_vstack(list_of_vectors: list[tf.Tensor]) -> tf.Tensor:
    """Tensors should be unidimensional"""
    return tf.transpose(tf.stack(list_of_vectors))


def tf_append(tensor: tf.Tensor, vector: tf.Tensor):
    """Append vector (1d) to the end of tensor"""
    return tf.concat([tensor, vector[:, None]], axis=1)


class TensorExpress:
    def __init__(self, tensor):
        self._tensor = tensor

    @property
    def tensor(self):
        return self._tensor

    def __getattr__(self, name):
        results = getattr(self._tensor, name)
        if isinstance(results, tf.Tensor):
            return TensorExpress(results)
        else:
            return results

    def __getitem__(self, item):
        return self.tensor.__getitem__(item)

    def astype(self, dtype):
        return TensorExpress(tf.cast(self._tensor, dtype=dtype))

    def long(self):
        return self.astype(tf.float32)

    def int(self):
        return self.astype(tf.int32)

    def __repr__(self):
        return self._tensor.__repr__()

    def all(self, axis=None):
        return TensorExpress(tf.reduce_all(self._tensor, axis=axis))

    def any(self, axis=None):
        return TensorExpress(tf.reduce_any(self._tensor, axis=axis))

    def sum(self, axis=None):
        return TensorExpress(tf.reduce_sum(self._tensor, axis=axis))
