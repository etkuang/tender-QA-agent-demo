# coding: utf-8
# @Author: Wang Qingkang

import json
import requests
import streamlit as st

from app_backend_layer.core.config import settings
from app_backend_layer.core.logger import frontend_logger
from app_frontend_layer.components.sidebar import refresh_history

BACKEND_URL = str(settings.BACKEND_URL).rstrip("/")


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

        # Track if this is the initial turn of a new session
        is_first_turn = False
        if st.session_state.session_title == "New Chat":
            st.session_state.session_title = prompt[:20] + "..."
            is_first_turn = True

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Initialize dynamic UI components for streaming
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
                with requests.post(f"{BACKEND_URL}/chat/stream", json=payload, stream=True,
                                   timeout=settings.STREAM_TIMEOUT) as response:
                    response.raise_for_status()

                    # Intercept and route NDJSON streams using line-buffered reading
                    for line in response.iter_lines(decode_unicode=True):
                        if not line:
                            continue

                        try:
                            match json.loads(line):
                                case {"type": "thinking", "content": str(chunk_content)}:
                                    full_reasoning += chunk_content
                                    # Lazy initialization
                                    if status_container is None:
                                        status_container = st.status("🧠 Model is thinking...", expanded=True)
                                        status_placeholder = status_container.empty()
                                    status_placeholder.markdown(full_reasoning + "▌")
                                case {"type": "text", "content": str(chunk_content)}:
                                    # Finalize thinking container
                                    if status_container is not None and getattr(status_container, "_state", "") != "complete":
                                        status_placeholder.markdown(full_reasoning)
                                        status_container.update(label="🧠 Show Thinking", state="complete", expanded=False)
                                    full_response += chunk_content
                                    response_placeholder.markdown(full_response + "▌")
                                case unexpected_payload:
                                    frontend_logger.warning(f"Unrecognized stream payload schema skipped: {unexpected_payload}")
                        except json.JSONDecodeError as e:
                            frontend_logger.warning(f"NDJSON Parse Failure | Error: {e} | Malformed Fragment: {repr(line[:200])}...")

                # Stream complete cleanup
                response_placeholder.markdown(full_response)

            except Exception as e:
                st.error(f"Streaming Pipeline Error: {str(e)}")

            # Update local state memory
            st.session_state.messages.append({"type": "human", "data": {"content": prompt}})
            if full_reasoning:
                st.session_state.messages.append({"type": "reasoning", "data": {"content": full_reasoning}})
            st.session_state.messages.append({"type": "ai", "data": {"content": full_response}})

            # Invalidate stale history and trigger complete UI reconciliation
            if is_first_turn:
                refresh_history()
                st.rerun()
