# coding: utf-8
# @Author: Wang Qingkang

import json
import uuid
from functools import wraps

import requests
from requests.exceptions import ConnectionError, HTTPError, RequestException, Timeout

from common.logger import get_logger, reset_request_id, set_request_id
from app_frontend_layer.config import settings

BACKEND_URL = settings.BACKEND_URL.rstrip("/")
RETRYABLE_STATUS_CODES = {502, 503, 504}
logger = get_logger("frontend.api_client")


def request_success(data=None) -> dict:
    return {"ok": True, "data": data}


def request_error(request_id: str, retryable: bool = False) -> dict:
    return {"ok": False, "request_id": request_id, "retryable": retryable}


def with_request_context(func):
    @wraps(func)
    def wrapper(*args, **kwargs) -> dict | None:
        request_id = uuid.uuid4().hex
        token = set_request_id(request_id)
        try:
            return func(*args, request_id=request_id, **kwargs)
        except (ConnectionError, Timeout):
            logger.warning("%s network error", func.__name__, exc_info=True)
            return request_error(request_id, retryable=True)
        except HTTPError as err:
            response = err.response
            status_code = response.status_code if response is not None else None
            retryable = status_code in RETRYABLE_STATUS_CODES
            logger.warning(
                "%s HTTP error | request_id=%s | status_code=%s | retryable=%s",
                func.__name__,
                request_id,
                status_code,
                retryable,
            )
            return request_error(request_id, retryable=retryable)
        except RequestException:
            logger.warning("%s request error", func.__name__, exc_info=True)
            return request_error(request_id)
        finally:
            reset_request_id(token)

    return wrapper


@with_request_context
def request_session_messages(session_id: str, *, request_id: str) -> dict:
    response = requests.get(
        f"{BACKEND_URL}/sessions/{session_id}/messages",
        timeout=settings.INTERNAL_TIMEOUT,
        headers={"X-Request-ID": request_id},
    )
    response.raise_for_status()
    return request_success(response.json())


@with_request_context
def request_history_list(*, request_id: str) -> dict:
    response = requests.get(
        f"{BACKEND_URL}/sessions/get_list",
        timeout=settings.INTERNAL_TIMEOUT,
        headers={"X-Request-ID": request_id},
    )
    response.raise_for_status()
    return request_success(response.json())


@with_request_context
def request_delete_session(session_id: str, *, request_id: str) -> dict:
    response = requests.delete(
        f"{BACKEND_URL}/sessions/{session_id}",
        timeout=settings.INTERNAL_TIMEOUT,
        headers={"X-Request-ID": request_id},
    )
    response.raise_for_status()
    return request_success(True)


@with_request_context
def request_chat_stream(payload: dict, on_payload, *, request_id: str) -> dict:
    with requests.post(
        f"{BACKEND_URL}/chat/stream",
        json=payload,
        stream=True,
        timeout=settings.STREAM_TIMEOUT,
        headers={"X-Request-ID": request_id},
    ) as response:
        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):
            try:
                on_payload(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("NDJSON parse failure | malformed_fragment=%r", line[:200], exc_info=True)

    return request_success(True)