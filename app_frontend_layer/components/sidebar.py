# coding: utf-8
# @Author: Wang Qingkang

import uuid

import requests
import streamlit as st
from requests.exceptions import ConnectionError, RequestException, Timeout

from common.logging import get_logger, reset_request_id, set_request_id
from app_frontend_layer.components.settings import render_settings_dialog
from app_frontend_layer.config import settings


BACKEND_URL = str(settings.BACKEND_URL).rstrip("/")
logger = get_logger("frontend.sidebar")


def error_id_from_response(response, fallback: str) -> str:
    if response is None:
        return fallback
    return response.headers.get("X-Request-ID", fallback)


def fetch_session_messages(session_id: str):
    """ Fetch historical messages for a given session from SQLite via REST API."""
    request_id = uuid.uuid4().hex
    token = set_request_id(request_id)
    try:
        resp = requests.get(
            f"{BACKEND_URL}/sessions/{session_id}/messages",
            timeout=settings.INTERNAL_TIMEOUT,
            headers={"X-Request-ID": request_id},
        )
    except (ConnectionError, Timeout):
        logger.warning("Failed to fetch session messages due to network error | session_id=%s", session_id, exc_info=True)
        st.error(f"网络连接失败，请检查网络后重试。错误编号：{request_id}")
        return []
    except RequestException:
        logger.warning("Failed to fetch session messages | session_id=%s", session_id, exc_info=True)
        st.error(f"会话加载失败，系统已记录错误日志。错误编号：{request_id}")
        return []
    finally:
        reset_request_id(token)

    if resp.status_code == 200:
        return resp.json()

    error_id = error_id_from_response(resp, request_id)
    logger.warning("Failed to fetch session messages | session_id=%s | status_code=%s", session_id, resp.status_code)
    st.error(f"会话加载失败，系统已记录错误日志。错误编号：{error_id}")
    return []


def load_session(session_id: str, title: str):
    """ Switch current chat context to the selected session."""
    st.session_state.session_id = session_id
    st.session_state.session_title = title
    st.session_state.messages = fetch_session_messages(session_id)


def refresh_history():
    """ Retrieve and update the global history metadata list."""
    try:
        history_resp = requests.get(f"{BACKEND_URL}/sessions/get_list", timeout=settings.INTERNAL_TIMEOUT)
    except RequestException:
        logger.warning("Failed to refresh history list", exc_info=True)
        st.session_state.history_list = []
        return

    if history_resp.status_code == 200:
        st.session_state.history_list = history_resp.json()
        return

    logger.warning("Failed to refresh history list | status_code=%s", history_resp.status_code)
    st.session_state.history_list = []


@st.fragment
def render_history_list():
    """ Renders the interactive list of historical chat sessions."""
    st.subheader("Chat History")
    refresh_history()
    for hist in st.session_state.history_list:
        col1, col2 = st.columns([5, 1])
        with col1:
            is_active = (hist["session_id"] == st.session_state.session_id)
            if st.button(
                f"💬 {hist['title']}",
                key=f"btn_{hist['session_id']}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                load_session(hist["session_id"], hist["title"])
                st.rerun()

        with col2:
            if st.button("🗑️", key=f"del_{hist['session_id']}"):
                request_id = uuid.uuid4().hex
                token = set_request_id(request_id)
                try:
                    delete_resp = requests.delete(
                        f"{BACKEND_URL}/sessions/{hist['session_id']}",
                        timeout=settings.INTERNAL_TIMEOUT,
                        headers={"X-Request-ID": request_id},
                    )
                except (ConnectionError, Timeout):
                    logger.warning("Delete session network error | session_id=%s", hist["session_id"], exc_info=True)
                    st.error(f"网络连接失败，请检查网络后重试。错误编号：{request_id}")
                    return
                except RequestException:
                    logger.warning("Delete session request error | session_id=%s", hist["session_id"], exc_info=True)
                    st.error(f"删除失败，系统已记录错误日志。错误编号：{request_id}")
                    return
                finally:
                    reset_request_id(token)

                error_id = error_id_from_response(delete_resp, request_id)
                if delete_resp.status_code in {502, 503, 504}:
                    logger.warning(
                        "Delete session retryable service error | session_id=%s | status_code=%s",
                        hist["session_id"],
                        delete_resp.status_code,
                    )
                    st.error(f"服务暂时不可用，请稍后重试。错误编号：{error_id}")
                    return

                if delete_resp.status_code != 200:
                    logger.warning(
                        "Delete session failed | session_id=%s | status_code=%s",
                        hist["session_id"],
                        delete_resp.status_code,
                    )
                    st.error(f"删除失败，系统已记录错误日志。错误编号：{error_id}")
                    return

                if hist["session_id"] == st.session_state.session_id:
                    del st.session_state.app_initialized
                    st.rerun()

                st.rerun(scope="fragment")


def render_sidebar():
    """ Renders the primary navigation sidebar."""
    with st.sidebar:
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.session_title = "New Chat"
            st.session_state.messages = []
            st.rerun()

        st.divider()
        if st.button("⚙️ Settings", use_container_width=True):
            render_settings_dialog()

        st.divider()
        render_history_list()
