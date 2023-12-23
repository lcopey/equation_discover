from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.optimize import basinhopping
from tensorflow.keras.losses import MSE
from tensorflow.keras.models import Model

from .tokens import BASE_TOKENS, Token, TokenSequence


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
            operator=token.tensorflow,
            parent=parent,
            left_child=left_child,
            right_child=right_child,
            variable=token.variable,
            constant=(not token.variable) & (token.arity == 0),
        )

    @classmethod
    def from_sequence(cls, sequence, tokens: TokenSequence = BASE_TOKENS):
        value = sequence[0]
        token = tokens[value]
        root = cls.from_token(token, parent=None)

        current_arity = root.arity
        previous = root

        for value in sequence[1:]:
            current_arity -= 1
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
        case other:
            return f"{node.symbol}"


def latex_repr(node: "Node", left_repr: str, right_repr: str, level: int = 0):
    match node.symbol:
        case "/":
            return f"\\frac{{{left_repr}}}{{{right_repr}}}"
        case "exp":
            return f"e^{{{left_repr}}}"
        case other:
            match node.arity:
                case 2:
                    return f"{left_repr}{node.symbol}{right_repr}"
                case 1:
                    return f"{node.symbol} ({left_repr})"
                case other:
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


def format_as_dict_of_tensor(X: pd.DataFrame):
    return {
        key: tf.convert_to_tensor(value, dtype=tf.float32)
        for key, value in X.to_dict("list").items()
    }


def tf_eval(
    node: "Node",
    left,
    right,
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


class Expression:
    def __init__(self, tree: "Node"):
        self.tree = tree
        self.n_const = tree.n_constant
        self.constants = np.random.randn(self.n_const)

    @staticmethod
    def preprocess(X: pd.DataFrame):
        if isinstance(X, pd.DataFrame):
            return format_as_dict_of_tensor(X)
        return X

    def eval(self, X: pd.DataFrame, constants: Optional[Iterable] = None):
        X = self.preprocess(X)
        if constants is None:
            constants = self.constants

        return self.tree.tf_eval(X=X, constants=constants)

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        T=1e-3,
        step_size=1e-1,
        niter: int = 500,
        **kwargs,
    ):
        # TODO optimize parameters :
        #  T, stepsize, niter, nitersucess
        #  see to implement BFGS in tensorflow ?
        res = basinhopping(
            lambda constants: MSE(y, self.eval(X, constants)),
            self.constants,
            T=T,
            stepsize=step_size,
            niter=niter,
            **kwargs,
        )
        self.res_ = res
        self.constants = res.x
        return self


class EvalModel(Model):
    def __init__(self, tree: "Node"):
        super().__init__()
        self.tree = tree
        num_constants = tree.n_constant
        self.constants = tf.Variable(
            np.random.randn(num_constants) * 4 - 2, dtype=tf.float32, trainable=True
        )
        self.build(tf.TensorShape(None))

    def call(self, inputs: Any, training: Any = None, mask: Any = None):
        return self.tree.tf_eval(X=inputs, constants=self.constants)

    @wraps(Model.fit)
    def fit(self, x, y, *args, **kwargs):
        if isinstance(x, pd.DataFrame):
            x = format_as_dict_of_tensor(x)
        if isinstance(y, pd.Series):
            y = tf.convert_to_tensor(y)

        super().fit(x, y, *args, **kwargs)
