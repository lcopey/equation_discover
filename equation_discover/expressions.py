from dataclasses import dataclass
from typing import Callable, Optional

from .tokens import Token


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
    def build_tree(cls, sequence):
        pass

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

    def remaining_children(self):
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
    def _child_tree_repr(node: Optional["Node"], level: int = 0):
        tab = ("   " * level) + " |-"
        return f"{tab}{node._recursive_tree_repr(level=level + 1)}" if node else None

    def _recursive_tree_repr(self, level: int = 0):
        left_repr = self._child_tree_repr(self.left_child, level=level)
        right_repr = self._child_tree_repr(self.right_child, level=level)

        return "\n".join(
            filter(lambda x: x is not None, (self.symbol, left_repr, right_repr))
        )

    @staticmethod
    def _child_expr_repr(node: Optional["Node"]):
        return node._recursive_expr_repr() if node else None

    def _recursive_expr_repr(self):
        match self.arity:
            case 2:
                return f"{self._child_expr_repr(self.left_child)} {self.symbol} {self._child_expr_repr(self.right_child)}"
            case 1:
                return f"{self.symbol}({self._child_expr_repr(self.left_child)})"
            case other:
                return self.symbol

    @staticmethod
    def _child_html_repr(node: Optional["Node"], level: int = 0):
        return f"{node._recursive_html_repr(level=level + 1)}" if node else None

    def _recursive_html_repr(self, level: int = 0):
        left_repr = self._child_html_repr(self.left_child, level=level)
        right_repr = self._child_html_repr(self.right_child, level=level)
        match self.arity:
            case 2:
                return (
                    f"<li>"
                    f"<details open>"
                    f"<summary>{self.symbol}</summary>"
                    f"\t<ul>{left_repr}</ul>"
                    f"\t<ul>{right_repr}</ul>"
                    f"</details>"
                    f"</li>"
                )
            case 1:
                return (
                    f"<li>"
                    f"<details open>"
                    f"<summary>{self.symbol}</summary>"
                    f"\t<ul>{left_repr}</ul>"
                    f"</details>"
                    f"</li>"
                )
            case other:
                return f"<li>{self.symbol}</li>"

    def __repr__(self):
        # return self._recursive_tree_repr()
        return self._recursive_expr_repr()

    def _repr_html_(self):
        template = "<div>" f"\t{self._recursive_html_repr()}" "</div>"
        return template
