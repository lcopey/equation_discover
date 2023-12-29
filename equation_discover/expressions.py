from dataclasses import dataclass
from functools import wraps
from typing import (Any, Callable, Iterable, Literal, Optional, TypedDict,
                    overload)

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.optimize import basinhopping
from tensorflow.keras.losses import MSE
from tensorflow.keras.models import Model

from .constants import TF_FLOAT_DTYPE
from .logger import getLogger
from .rewards import rsquared
from .tokens import BASE_TOKENS, Token, TokenLibrary

LOGGER = getLogger("Expression")


@dataclass
class Node:
    symbol: str
    arity: float
    operator: Callable
    parent: Optional["Node"] = None
    left_child: Optional["Node"] = None
    right_child: Optional["Node"] = None
    variable: Optional[bool] = False
    constant: Optional[bool] = False

    @classmethod
    def from_token(
        cls,
        token: Token,
        parent: Optional["Node"] = None,
        left_child: Optional["Node"] = None,
        right_child: Optional["Node"] = None,
    ):
        return cls(
            symbol=token.symbol,
            arity=token.arity,
            operator=token.tf_eval,
            parent=parent,
            left_child=left_child,
            right_child=right_child,
            variable=token.variable,
            constant=(not token.variable) & (token.arity == 0),
        )

    @classmethod
    def from_sequence(cls, sequence, tokens: TokenLibrary = BASE_TOKENS):
        value = sequence[0]
        token = tokens[value]
        root = cls.from_token(token, parent=None)

        current_arity = root.arity
        previous = root

        for value in sequence[1:]:
            current_arity -= 1.0
            token = tokens[value]
            child = Node.from_token(token)
            current_arity += child.arity

            previous.add_child(child)

            previous = child
            while not previous.is_complete():
                previous = previous.parent
                if previous is None:
                    break
        return root

    @property
    def is_final(self):
        return self.arity == 0

    @property
    def is_unary(self):
        return self.arity == 1

    @property
    def is_binary(self):
        return self.arity == 2

    @property
    def child_count(self) -> int:
        return sum((self.left_child is not None, self.right_child is not None))

    @property
    def n_constant(self) -> int:
        return walk(self, count_constant)

    def tf_eval(self, X: pd.DataFrame, constants: Optional[Iterable] = None):
        if constants is None:
            constants = []
        X = pandas_to_tensor(X)
        return walk(self, tf_eval, X=X, constants={"iter": 0, "value": constants})

    def is_complete(self):
        return self.child_count < self.arity

    def set_parent(self, parent: "Node"):
        self.parent = parent

    def add_child(self, node: "Node"):
        node.parent = self
        if self.left_child is None:
            self.left_child = node
        elif self.right_child is None:
            self.right_child = node
        else:
            raise KeyError

    @staticmethod
    def recursive_repr(
        node: "Node",
        repr_func: Callable[["Node", str, str, int], str],
        level: int = 0,
    ):
        left_repr = (
            Node.recursive_repr(node.left_child, repr_func, level + 1)
            if node.left_child
            else None
        )
        right_repr = (
            Node.recursive_repr(node.right_child, repr_func, level + 1)
            if node.right_child
            else None
        )

        return repr_func(node, left_repr, right_repr, level)

    def __repr__(self):
        return self.recursive_repr(self, standard_node_repr, 0)

    # def _repr_html_(self):
    #     template = "<div>" f"\t{self.recursive_repr(self, html_node_repr, 0)}" "</div>"
    #     return template

    def _repr_latex_(self):
        return f"$${self.recursive_repr(self, latex_repr, 0)}$$"


def html_node_repr(node: "Node", left_repr, right_repr, level: int = 0):
    match node.arity:
        case 2:
            return (
                f"<li>"
                f"<details open>"
                f"<summary>{node.symbol}</summary>"
                f"\t<ul>{left_repr}</ul>"
                f"\t<ul>{right_repr}</ul>"
                f"</details>"
                f"</li>"
            )
        case 1:
            return (
                f"<li>"
                f"<details open>"
                f"<summary>{node.symbol}</summary>"
                f"\t<ul>{left_repr}</ul>"
                f"</details>"
                f"</li>"
            )
        case other:
            return f"<li>{node.symbol}</li>"


def standard_node_repr(node: "Node", left_repr: str, right_repr: str, level: int = 0):
    match node.arity:
        case 2:
            return f"{left_repr} {node.symbol} {right_repr}"
        case 1:
            return f"{node.symbol}({left_repr})"
        case other:
            return node.symbol


def tree_node_repr(node: "Node", left_repr: str, right_repr: str, level: int = 0):
    tab = "  " * level
    match node.arity:
        case 2:
            return f"{node.symbol}\n" f"{tab}|-{left_repr}\n" f"{tab}|-{right_repr}"
        case 1:
            return f"{node.symbol}\n" f"{tab}|-{left_repr}\n"
        case _:
            return f"{node.symbol}"


def latex_repr(node: "Node", left_repr: str, right_repr: str, _: int = 0):
    match node.symbol:
        case "/":
            return f"\\frac{{{left_repr}}}{{{right_repr}}}"
        case "exp":
            return f"e^{{{left_repr}}}"
        case _:
            match node.arity:
                case 2:
                    return f"{left_repr}{node.symbol}{right_repr}"
                case 1:
                    return f"{node.symbol} ({left_repr})"
                case _:
                    return node.symbol


def walk(node: "Node", callback: Callable, **kwargs):
    if node.arity != 0:
        left = walk(node.left_child, callback, **kwargs)
        if node.right_child is not None:
            right = walk(node.right_child, callback, **kwargs)
        else:
            right = None
        return callback(node, left, right, **kwargs)

    else:
        return callback(node, None, None, **kwargs)


def count_constant(
    node: "Node", left: Optional[int], right: Optional[int], current_constant: int = 0
) -> int:
    count = 0
    if left is not None:
        count += left

    if right is not None:
        count += right

    if node.constant:
        return current_constant + 1 + count
    else:
        return current_constant + count


def tf_eval(
    node: "Node",
    left: Optional[tf.Tensor],
    right: Optional[tf.Tensor],
    X: dict[str, tf.Tensor],
    constants: dict[str, tf.Tensor | int],
):
    if node.constant:
        current_iter = constants["iter"]
        constants["iter"] = current_iter + 1
        return constants["value"][current_iter]
    if node.variable:
        return X[node.symbol]
    if node.arity == 2:
        return node.operator(left, right)
    if node.arity == 1:
        return node.operator(left)

    raise


def make_val_and_grad_fn(value_fn):
    @wraps(value_fn)
    def val_and_grad(x):
        return tfp.math.value_and_gradient(value_fn, x)

    return val_and_grad


class ExpressionInputs(TypedDict):
    X: pd.DataFrame | dict[str, tf.Tensor]
    constants: Optional[Iterable]


@overload
def pandas_to_tensor(value: pd.DataFrame) -> dict[str, tf.Tensor]:
    ...


@overload
def pandas_to_tensor(value: pd.Series) -> tf.Tensor:
    ...


@overload
def pandas_to_tensor(value: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
    ...


@overload
def pandas_to_tensor(value: tf.Tensor) -> tf.Tensor:
    ...


def pandas_to_tensor(
    value: pd.DataFrame | pd.Series | dict[str, tf.Tensor] | tf.Tensor,
) -> dict[str, tf.Tensor] | tf.Tensor:
    if isinstance(value, pd.DataFrame):
        return {
            key: tf.convert_to_tensor(value, dtype=TF_FLOAT_DTYPE)
            for key, value in value.to_dict("list").items()
        }
    elif isinstance(value, pd.Series):
        return tf.convert_to_tensor(value, dtype=TF_FLOAT_DTYPE)

    else:
        return value


class Expression(Model):
    def __init__(self, tree: "Node"):
        super().__init__()
        self.tree = tree
        self.n_const = tree.n_constant
        self.constants = tf.Variable(
            np.random.randn(self.n_const) * 4 - 2, dtype=TF_FLOAT_DTYPE, trainable=True
        )
        self.build(tf.TensorShape(None))

    def call(self, inputs: ExpressionInputs, training: Any = None, mask: Any = None):
        X = inputs["X"]
        constants = inputs["constants"]
        return self.tree.tf_eval(X=X, constants=constants)

    def eval(
        self,
        X: pd.DataFrame | dict[str, tf.Tensor],
        constants: Optional[Iterable] = None,
    ):
        if constants is None:
            constants = self.constants
        X = pandas_to_tensor(X)
        return self({"X": X, "constants": constants})

    def optimize_constants(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        mode: Literal["lbfgs", "basinhopping"],
        **kwargs,
    ):
        X = pandas_to_tensor(X)
        y = pandas_to_tensor(y)
        if self.n_const > 0:
            if mode == "lbfgs":
                return self._lbfgs(X, y, **kwargs)
            elif mode == "basinhopping":
                return self._basinhopping(X, y, **kwargs)
        else:
            self.res_ = "No const to optimize"

    def _lbfgs(self, X: dict[str, tf.Tensor], y: tf.Tensor = None, *args, **kwargs):
        value_and_grad_func = make_val_and_grad_fn(
            lambda constants: MSE(y, self.eval(X, constants))
        )
        res = tfp.optimizer.lbfgs_minimize(value_and_grad_func, self.constants)
        self.res_ = res
        self.constants = self.res_.position
        return self

    def _basinhopping(
        self,
        X: dict[str, tf.Tensor],
        y: tf.Tensor = None,
        T: float = 1e-3,
        step_size: float = 0.1,
        niter: int = 500,
        **kwargs,
    ):
        # TODO optimize parameters :
        #  T, stepsize, niter, nitersucess
        y = tf.convert_to_tensor(y)
        res = basinhopping(
            lambda constants: MSE(y, self.eval(X, constants)),
            np.array(self.constants),
            T=T,
            stepsize=step_size,
            niter=niter,
            # callback=print,
            **kwargs,
        )
        self.res_ = res
        self.constants = res.x
        return self

    def reward(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        reward_func: Callable[[tf.Tensor, tf.Tensor], float] = None,
    ):
        if reward_func is None:
            reward_func = rsquared
        y_pred = self.eval(X)
        y = pandas_to_tensor(y)
        return reward_func(y, y_pred)


class ExpressionEnsemble(list[Expression]):
    def __init__(
        self,
        sequences: tf.Tensor,
        lengths: tf.Tensor,
    ):
        list_of_expressions = [
            Expression(Node.from_sequence(sequence[:length]))
            for sequence, length in zip(sequences, lengths)
        ]
        super().__init__(list_of_expressions)
        self.logger = LOGGER.bind(object="ExpressionEnsemble")

    def eval(self, X: pd.DataFrame):
        """Returns the evaluation of all the expressions (n_expression, n_point)"""
        return tf.stack([expression.eval(X) for expression in self])

    def optimize_constants(
        self, X: pd.DataFrame | dict[str, tf.Tensor], y: pd.Series | tf.Tensor, **kwargs
    ):
        for n, expression in enumerate(self):
            self.logger.debug(f"Fitting {n + 1}/{len(self)}: {expression.tree}")
            expression.optimize_constants(X, y, mode="lbfgs", **kwargs)

    def score(
        self,
        X: pd.DataFrame | dict[str, tf.Tensor],
        y: pd.Series | tf.Tensor,
        score_func: Callable[[tf.Tensor, tf.Tensor], float] = None,
    ):
        if score_func is None:
            score_func = rsquared
        y = pandas_to_tensor(y)
        y_pred = self.eval(X)
        return score_func(y_pred, y)
