# coding: utf-8
# @Author: Wang Qingkang

import contextvars
import logging


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


def get_logger(name: str) -> logging.Logger:
    _configure_logging_once()
    return logging.getLogger(name)


def _configure_logging_once() -> None:
    global _configured
    if _configured:
        return

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in root.handlers:
        root.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] [req=%(request_id)s] %(name)s: %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
    )
    handler.addFilter(RequestIdFilter())
    root.addHandler(handler)
    _configured = True


frontend_logger = get_logger("FrontendStreamParser")
backend_stream_logger = get_logger("BackendStreamPipeline")
