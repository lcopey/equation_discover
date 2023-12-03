from dataclasses import dataclass
from typing import Callable

import tensorflow as tf


@dataclass
class Token:
    symbol: str
    arity: float
    tensorflow: Callable | None = None
    sympy: Callable | None = None
    variable: bool = False

    def __repr__(self):
        return self.symbol


class TokenSequence(list[Token]):
    def __init__(self, *iterable: Token):
        super().__init__(iterable)
        self._dict_map = {token.symbol: token for token in iterable}

    def __getitem__(self, item: int | slice | str):
        if isinstance(item, str):
            return self._dict_map[item]
        else:
            return super().__getitem__(item)

    @property
    def nonvariable_mask(self):
        return tf.constant([not token.variable for token in self])

    @property
    def variable_mask(self):
        return tf.constant([token.variable for token in self])

    def i_arity_mask(self, i: int):
        return tf.constant([token.arity == i for token in self])

    def non_i_arity_mask(self, i: int):
        return tf.constant([token.arity != i for token in self])

    @property
    def zero_arity_mask(self):
        return self.i_arity_mask(0)

    @property
    def one_arity_mask(self):
        return self.i_arity_mask(1)

    @property
    def two_arity_mask(self):
        return self.i_arity_mask(2)

    @property
    def non_zero_arity_mask(self):
        return self.non_i_arity_mask(0)

    @property
    def non_one_arity_mask(self):
        return self.non_i_arity_mask(1)

    @property
    def non_two_arity_mask(self):
        return self.non_i_arity_mask(2)

    @property
    def variable_tensor(self):
        return tf.constant([n for n, token in enumerate(self) if token.variable])

    def i_arity_tensor(self, i: int):
        return tf.constant([n for n, token in enumerate(self) if token.arity == i])

    def non_i_arity_tensor(self, i: int):
        return tf.constant([n for n, token in enumerate(self) if token.arity != i])

    @property
    def zero_arity_tensor(self):
        return self.i_arity_tensor(0)

    @property
    def one_arity_tensor(self):
        return self.i_arity_tensor(1)

    @property
    def two_arity_tensor(self):
        return self.i_arity_tensor(2)

    @property
    def non_zero_arity_tensor(self):
        return self.non_i_arity_tensor(0)

    @property
    def non_one_arity_tensor(self):
        return self.non_i_arity_tensor(1)

    @property
    def non_two_arity_tensor(self):
        return self.non_i_arity_tensor(2)


BASE_TOKENS = TokenSequence(
    *[
        Token(char, 2, func)
        for char, func in zip("+-*/", (tf.add, tf.subtract, tf.multiply, tf.divide))
    ]
)

BASE_TOKENS.extend(
    [
        Token(symbol, 1, func)
        for symbol, func in zip(
            ("sin", "cos", "exp", "log"), (tf.sin, tf.cos, tf.exp, tf.math.log)
        )
    ]
)
BASE_TOKENS.append(Token("const", 0))
BASE_TOKENS.append(Token("var_x", 0, variable=True))
