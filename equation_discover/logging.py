import logging
from typing import Literal

logging.basicConfig(
    format="%(process)d - %(levelname)s - %(asctime)s - %(message)s",
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

    def debug(self, msg: str, **kwargs):
        log(self.logger, msg, "debug", **self.kwargs, **kwargs)

    def info(self, msg: str, **kwargs):
        log(self.logger, msg, "info", **self.kwargs, **kwargs)

    def warning(self, msg: str, **kwargs):
        log(self.logger, msg, "warning", **self.kwargs, **kwargs)

    def error(self, msg: str, **kwargs):
        log(self.logger, msg, "error", **self.kwargs, **kwargs)

    def critical(self, msg: str, **kwargs):
        log(self.logger, msg, "critical", **self.kwargs, **kwargs)


def getLogger(name: str, **kwargs):
    return KeywordsLogger(name, **kwargs)


def log(
    logger: logging.Logger,
    msg,
    level: Literal["debug", "info", "warning", "error", "critical"],
    **kwargs,
):
    keywords = ", ".join(["=".join(item) for item in kwargs.items()])
    msg = f"{msg}, {keywords}"

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
