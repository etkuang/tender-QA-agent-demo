# coding: utf-8
# @Author: Wang Qingkang

import contextvars
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"

_configured = False

# Request ids correlate logs from one HTTP request;
# ContextVar keeps the value isolated across concurrent async tasks instead of using a shared global value.
_request_id_ctx = contextvars.ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    """ Inject the context-local request id into each log record."""

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
    """ Configure logging once from environment variables LOG_LEVEL and LOG_SERVICE_NAME,
        then return the named logger."""
    _configure_logging_once()
    return logging.getLogger(name)


def _configure_logging_once() -> None:
    global _configured
    if _configured:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    service_name = os.environ["LOG_SERVICE_NAME"]

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [req=%(request_id)s] %(name)s: %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)
    # Clear handlers that other modules may have added to avoid duplicate log output.
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
