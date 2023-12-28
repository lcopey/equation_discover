import logging
from typing import Any, Literal, Optional

import tensorflow as tf

logging.basicConfig(
    format="%(name)s - %(levelname)s - %(asctime)s - %(message)s",
    level=logging.DEBUG,
    datefmt="%d-%b-%y %H:%M:%S",
)


class KeywordsLogger:
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.logger = logging.getLogger(name)
        self.kwargs = kwargs

    def bind(self, **kwargs):
        return KeywordsLogger(self.name, **self.kwargs, **kwargs)

    def debug(self, msg: Optional[str] = None, **kwargs):
        log(self.logger, msg, "debug", **self.kwargs, **kwargs)

    def info(self, msg: Optional[str] = None, **kwargs):
        log(self.logger, msg, "info", **self.kwargs, **kwargs)

    def warning(self, msg: Optional[str] = None, **kwargs):
        log(self.logger, msg, "warning", **self.kwargs, **kwargs)

    def error(self, msg: Optional[str] = None, **kwargs):
        log(self.logger, msg, "error", **self.kwargs, **kwargs)

    def critical(self, msg: Optional[str] = None, **kwargs):
        log(self.logger, msg, "critical", **self.kwargs, **kwargs)


def getLogger(name: str, **kwargs):
    return KeywordsLogger(name, **kwargs)


def log(
    logger: logging.Logger,
    msg: Optional[str] = None,
    level: Literal["debug", "info", "warning", "error", "critical"] = "info",
    return_line: bool = False,
    **kwargs,
):
    if return_line:
        keywords = (
            "\n"
            + "\n".join(["=".join(map(repr, item)) for item in kwargs.items()])
            + "\n"
        )
    else:
        keywords = ", ".join(["=".join(map(str, item)) for item in kwargs.items()])

    if msg is not None:
        msg = f"{msg}, {keywords}"
    else:
        msg = keywords

    match level:
        case "debug":
            logger.debug(msg)
        case "info":
            logger.info(msg)
        case "warning":
            logger.warning(msg)
        case "error":
            logger.error(msg)
        case "critical":
            logger.critical(msg)


class TensorRepr:
    def __init__(self, tensor: tf.Tensor, max_length: int = 4):
        self.tensor = tensor
        self.float_fmt = "{:.2f}"
        self.int_fmt = "{:.0f}"
        self.bool_fmt = "{:}"
        self.max_length = max_length

    def value_formatter(self, value: Any):
        if isinstance(value, tf.Tensor) and value.dtype in (tf.float32, tf.float64):
            return self.float_fmt.format(value)
        elif isinstance(value, tf.Tensor) and value.dtype in (tf.int32, tf.int64):
            return self.int_fmt.format(value)
        elif isinstance(value, tf.Tensor) and value.dtype == tf.bool:
            return self.bool_fmt.format(value)
        else:
            return str(value)

    def get_items(self, items: tf.Tensor):
        if items.shape[0] > self.max_length:
            items = [
                *items[: self.max_length // 2],
                "...",
                *items[-self.max_length // 2 :],
            ]
        else:
            items = items
        return items

    def unidimensional_repr(self, items: tf.Tensor) -> str:
        items = self.get_items(items)
        return f"[{', '.join(map(self.value_formatter, items))}]"

    def tensor_repr(self, tensor: tf.Tensor, level: int = 0, n: int = 0) -> str:
        if tensor.shape.rank == 1:
            spacer = " " * (n - level) if n == 0 else " " * level
            return f"{spacer}{self.unidimensional_repr(tensor)}"

        else:
            items = self.get_items(tensor)
            this_level_repr = []
            for n, item in enumerate(items):
                if isinstance(item, tf.Tensor):
                    this_level_repr.append(self.tensor_repr(item, level=level + 1, n=n))
                else:
                    spacer = "" if n == 0 else " " * (n + level - 1)
                    this_level_repr.append(f"{spacer}{item}")

            return "[" + "\n".join(this_level_repr) + "]"

    def __repr__(self):
        return f"<Tensor shape: {self.tensor.shape}\n{self.tensor_repr(tensor=self.tensor)}>"


def repr(item: Any):
    if isinstance(item, tf.Tensor):
        return str(TensorRepr(item))
    else:
        return str(item)
