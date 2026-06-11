# coding: utf-8
# @Author: Wang Qingkang

import uuid

import streamlit as st

from app_frontend_layer.api_client import request_delete_session, request_history_list, request_session_messages


def fetch_session_messages(session_id: str):
    """Fetch historical messages for a given session from SQLite via REST API."""
    result = request_session_messages(session_id)
    if result["ok"]:
        return result["data"]

    if result["retryable"]:
        st.error(f"服务暂时不可用，请稍后重试。错误编号：{result['request_id']}")
    else:
        st.error(f"会话加载失败，系统已记录错误日志。错误编号：{result['request_id']}")
    return []


def load_session(session_id: str, title: str):
    """ Switch current chat context to the selected session."""
    st.session_state.session_id = session_id
    st.session_state.session_title = title
    st.session_state.messages = fetch_session_messages(session_id)


def refresh_history():
    """Retrieve and update the global history metadata list."""
    result = request_history_list()
    if result["ok"]:
        st.session_state.history_list = result["data"]
        return

    st.session_state.history_list = []


@st.fragment
def render_history_list():
    """Renders the interactive list of historical chat sessions."""
    st.subheader("Chat History")
    refresh_history()
    for hist in st.session_state.history_list:
        col1, col2 = st.columns([5, 1])
        with col1:
            is_active = hist["session_id"] == st.session_state.session_id
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
                result = request_delete_session(hist["session_id"])
                if not result["ok"]:
                    if result["retryable"]:
                        st.error(f"服务暂时不可用，请稍后重试。错误编号：{result['request_id']}")
                    else:
                        st.error(f"删除失败，系统已记录错误日志。错误编号：{result['request_id']}")
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
        render_history_list()
