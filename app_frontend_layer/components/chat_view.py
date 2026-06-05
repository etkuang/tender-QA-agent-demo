# coding: utf-8
# @Author: Wang Qingkang

import json
import uuid

import requests
import streamlit as st
from requests.exceptions import ConnectionError, HTTPError, RequestException, Timeout

from common.logging import get_logger, reset_request_id, set_request_id
from app_frontend_layer.components.sidebar import refresh_history
from app_frontend_layer.config import settings

BACKEND_URL = str(settings.BACKEND_URL).rstrip("/")
frontend_logger = get_logger("frontend.chat")


def is_retryable_http_error(err: HTTPError) -> bool:
    response = err.response
    return response is not None and response.status_code in {502, 503, 504}


def error_id_from_response(response, fallback: str) -> str:
    if response is None:
        return fallback
    return response.headers.get("X-Request-ID", fallback)


def render_chat():
    """ Main rendering loop for conversational interface and SSE pipeline processing."""
    # 1. Render historical messages sequentially
    for msg in st.session_state.messages:
        msg_type = msg["type"]
        content = msg["data"]["content"]
        if msg_type == "reasoning":
            with st.chat_message("assistant"):
                with st.status("🧠 Show Thinking", state="complete", expanded=False):
                    st.markdown(content)
        else:
            role = "user" if msg_type in ["human", "user"] else "assistant"
            with st.chat_message(role):
                st.markdown(content)

    # 2. Input Pipeline and Inference Execution
    if prompt := st.chat_input("Ask tender-agent"):
        request_id = uuid.uuid4().hex
        token = set_request_id(request_id)

        # Track if this is the initial turn of a new session
        is_first_turn = False
        if st.session_state.session_title == "New Chat":
            st.session_state.session_title = prompt[:20] + "..."
            is_first_turn = True

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            status_container = None
            status_placeholder = None
            response_placeholder = st.empty()

            full_response = ""
            full_reasoning = ""

            payload = {
                "session_id": st.session_state.session_id,
                "session_title": st.session_state.session_title,
                "prompt": prompt,
                "temperature": st.session_state.temp,
                "top_p": st.session_state.top_p,
                "max_tokens": st.session_state.max_t,
            }

            try:
                with requests.post(
                    f"{BACKEND_URL}/chat/stream",
                    json=payload,
                    stream=True,
                    timeout=settings.STREAM_TIMEOUT,
                    headers={"X-Request-ID": request_id},
                ) as response:
                    response.raise_for_status()

                    # Intercept and route NDJSON streams using line-buffered reading.
                    for line in response.iter_lines(decode_unicode=True):
                        if not line:
                            continue

                        try:
                            match json.loads(line):
                                case {"type": "thinking", "content": str(chunk_content)}:
                                    full_reasoning += chunk_content
                                    if status_container is None:
                                        status_container = st.status("🧠 Model is thinking...", expanded=True)
                                        status_placeholder = status_container.empty()
                                    status_placeholder.markdown(full_reasoning + "▌")
                                case {"type": "text", "content": str(chunk_content)}:
                                    if status_container is not None and getattr(status_container, "_state", "") != "complete":
                                        status_placeholder.markdown(full_reasoning)
                                        status_container.update(label="🧠 Show Thinking", state="complete", expanded=False)
                                    full_response += chunk_content
                                    response_placeholder.markdown(full_response + "▌")
                                case {"type": "error", "content": str(error_content)}:
                                    frontend_logger.warning("Stream error received | content=%s", error_content)
                                    st.error(error_content)
                                case unexpected_payload:
                                    frontend_logger.warning("Unrecognized stream payload schema skipped: %s", unexpected_payload)
                        except json.JSONDecodeError:
                            frontend_logger.warning("NDJSON parse failure | malformed_fragment=%r", line[:200], exc_info=True)

                response_placeholder.markdown(full_response)

            except (ConnectionError, Timeout):
                frontend_logger.warning("Chat stream network error", exc_info=True)
                st.error(f"网络连接失败，请检查网络后重试。错误编号：{request_id}")
            except HTTPError as err:
                error_id = error_id_from_response(err.response, request_id)
                frontend_logger.warning("Chat stream HTTP error | error_id=%s", error_id, exc_info=True)
                if is_retryable_http_error(err):
                    st.error(f"服务暂时不可用，请稍后重试。错误编号：{error_id}")
                else:
                    st.error(f"本次回答生成失败，系统已记录错误日志。错误编号：{error_id}")
            except RequestException:
                frontend_logger.warning("Chat stream request error", exc_info=True)
                st.error(f"请求失败，系统已记录错误日志。错误编号：{request_id}")
            except Exception:
                frontend_logger.exception("Unexpected frontend chat error")
                st.error(f"界面处理失败，系统已记录错误日志。错误编号：{request_id}")
            finally:
                reset_request_id(token)

            st.session_state.messages.append({"type": "human", "data": {"content": prompt}})
            if full_reasoning:
                st.session_state.messages.append({"type": "reasoning", "data": {"content": full_reasoning}})
            if full_response:
                st.session_state.messages.append({"type": "ai", "data": {"content": full_response}})

            if is_first_turn:
                refresh_history()
                st.rerun()
