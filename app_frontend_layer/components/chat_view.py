# coding: utf-8
# @Author: Wang Qingkang

import streamlit as st

from common.api_contracts.backend_api import ChatRequest, StreamChunk
from app_frontend_layer.api_client import request_chat_stream
from app_frontend_layer.components.sidebar import refresh_history
from common.logger import get_logger

frontend_logger = get_logger("frontend.chat")


def render_chat():
    """ Main rendering loop for conversational interface and SSE pipeline processing."""
    # 1. Render historical messages sequentially
    for msg in st.session_state.messages:
        msg_type = msg["type"]
        content = msg["content"]
        if msg_type == "user":
            with st.chat_message("user"):
                st.markdown(content)
        elif msg_type == "assistant":
            with st.chat_message("assistant"):
                st.markdown(content)
        elif msg_type == "reasoning":
            with st.chat_message("assistant"):
                with st.status("🧠 Show Thinking", state="complete", expanded=False):
                    st.markdown(content)
        elif msg_type == "error":
            with st.chat_message("assistant"):
                st.error(content)

    # 2. Input Pipeline and Inference Execution
    if user_message := st.chat_input("Ask tender-agent"):
        # Track if this is the initial turn of a new session
        is_first_turn = False
        if st.session_state.session_title == "New Chat":
            st.session_state.session_title = user_message[:20] + "..."
            is_first_turn = True

        with st.chat_message("user"):
            st.markdown(user_message)

        with st.chat_message("assistant"):
            status_container = None
            status_placeholder = None
            response_placeholder = st.empty()

            full_response = ""
            full_reasoning = ""
            full_error = ""

            payload = ChatRequest(
                session_id=st.session_state.session_id,
                session_title=st.session_state.session_title,
                user_message=user_message,
            )

            def handle_stream_payload(stream_payload: StreamChunk) -> None:
                nonlocal full_response, full_reasoning, full_error, status_container, status_placeholder

                if stream_payload.type == "reasoning":
                    full_reasoning += stream_payload.content
                    if status_container is None:
                        status_container = st.status("🧠 Model is thinking...", expanded=True)
                        status_placeholder = status_container.empty()
                    status_placeholder.markdown(full_reasoning + "▌")
                elif stream_payload.type == "assistant":
                    if status_container is not None and getattr(status_container, "_state", "") != "complete":
                        status_placeholder.markdown(full_reasoning)
                        status_container.update(label="🧠 Show Thinking", state="complete", expanded=False)
                    full_response += stream_payload.content
                    response_placeholder.markdown(full_response + "▌")
                elif stream_payload.type == "error":
                    full_error += stream_payload.content
                    frontend_logger.warning("Stream error received | content=%s", stream_payload.content)
                    st.error(stream_payload.content)

            result = request_chat_stream(payload, handle_stream_payload)
            if result["ok"]:
                response_placeholder.markdown(full_response)
            elif result["retryable"]:
                st.error(f"服务暂时不可用，请稍后重试。错误编号：{result['request_id']}")
            else:
                st.error(f"本次回答生成失败，系统已记录错误日志。错误编号：{result['request_id']}")

            st.session_state.messages.append({"type": "user", "content": user_message})
            if full_reasoning:
                st.session_state.messages.append({"type": "reasoning", "content": full_reasoning})
            if full_response:
                st.session_state.messages.append({"type": "assistant", "content": full_response})
            if full_error:
                st.session_state.messages.append({"type": "error", "content": full_error})

            if is_first_turn:
                refresh_history()
                st.rerun()
