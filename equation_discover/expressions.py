from dataclasses import dataclass
from typing import Callable, Optional

from .tokens import BASE_TOKENS, Token, TokenSequence


@dataclass
class Node:
    symbol: str
    arity: float
    operator: Callable
    parent: Optional["Node"] = None
    left_child: Optional["Node"] = None
    right_child: Optional["Node"] = None

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
        )

    @classmethod
    def build_tree(cls, sequence, tokens: TokenSequence = BASE_TOKENS):
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
        return self.recursive_repr(self, latex_repr, 0)

    def _repr_html_(self):
        template = "<div>" f"\t{self.recursive_repr(self, html_node_repr, 0)}" "</div>"
        return template

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
