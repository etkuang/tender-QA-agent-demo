# coding: utf-8
# @Author: Wang Qingkang

import contextvars
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"

_request_id_ctx = contextvars.ContextVar("request_id", default="-")
_configured = False


class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = _request_id_ctx.get()
        return True


def set_request_id(request_id: str):
    return _request_id_ctx.set(request_id)


def reset_request_id(token) -> None:
    _request_id_ctx.reset(token)


def get_request_id() -> str:
    return _request_id_ctx.get()


def get_logger(name: str) -> logging.Logger:
    _configure_logging_once(name)
    return logging.getLogger(name)


def _configure_logging_once(default_service_name: str) -> None:
    global _configured
    if _configured:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    service_name = os.environ.get("LOG_SERVICE_NAME") or default_service_name.split(".", 1)[0]

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [req=%(request_id)s] %(name)s: %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)
    for handler in root.handlers:
        root.removeHandler(handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(RequestIdFilter())
    root.addHandler(stream_handler)

    file_handler = RotatingFileHandler(
        LOG_DIR / f"{service_name}.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(RequestIdFilter())
    root.addHandler(file_handler)

    _configured = True
