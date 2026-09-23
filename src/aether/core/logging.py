"""统一日志。

轻量封装标准库 logging，避免引入第三方依赖。
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "aether") -> logging.Logger:
    """返回带统一格式的 logger。"""

    global _CONFIGURED
    logger = logging.getLogger(name)
    if _CONFIGURED:
        return logger

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    _CONFIGURED = True
    return logger
