import numpy as np


class Bound:
    def sample(self, n: int):
        raise NotImplementedError

    @classmethod
    def from_array(cls, array: np.ndarray):
        raise NotImplementedError


class Uniform(Bound):
    def __init__(self, a, b, left_open=False):
        self.a = a
        self.b = b
        self.left_open = left_open

    @classmethod
    def from_array(cls, array: np.ndarray):
        a = array.min()
        b = array.max()
        return cls(a, b)

    def _sample(self, n: int) -> np.ndarray:
        return np.random.rand(n) * (self.b - self.a) + self.a

    def sample(self, n: int) -> np.ndarray:
        result = self._sample(n)
        if self.left_open:
            while result == self.a:
                result = self._sample(n)

        return result


class Bounds(list[Bound]):
    def __init__(self, bounds: list[Bound]):
        super().__init__(bounds)

    def sample(self, n: int) -> np.ndarray:
        return np.stack([bound.sample(n) for bound in self], -1)
