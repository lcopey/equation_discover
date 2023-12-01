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
    def zero_arity_mask(self):
        return tf.constant([token.arity == 0 for token in self])

    @property
    def nonzero_arity_mask(self):
        return tf.constant([token.arity != 0 for token in self])

    @property
    def nonvariable_mask(self):
        return tf.constant([not token.variable for token in self])

    @property
    def variable_mask(self):
        return tf.constant([token.variable for token in self])

    @property
    def variable_tensor(self):
        return tf.constant([n for n, token in enumerate(self) if token.variable])

    def arity_i_tensor(self, i: int):
        return tf.constant([n for n, token in enumerate(self) if token.arity == i])

    @property
    def arity_zero_tensor(self):
        return self.arity_i_tensor(0)

    @property
    def arity_one_tensor(self):
        return self.arity_i_tensor(1)

    @property
    def arity_two_tensor(self):
        return self.arity_i_tensor(2)


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
