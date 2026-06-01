# coding: utf-8
# @Author: Wang Qingkang

import uuid
import requests

import streamlit as st

from app_backend_layer.core.config import settings
from app_frontend_layer.components.settings import render_settings_dialog

BACKEND_URL = str(settings.BACKEND_URL).rstrip("/")


def fetch_session_messages(session_id: str):
    """ Fetch historical messages for a given session from SQLite via REST API."""
    resp = requests.get(f"{BACKEND_URL}/sessions/{session_id}/messages", timeout=settings.INTERNAL_TIMEOUT)
    if resp.status_code == 200:
        return resp.json()
    return None


def load_session(session_id: str, title: str):
    """ Switch current chat context to the selected session."""
    st.session_state.session_id = session_id
    st.session_state.session_title = title
    st.session_state.messages = fetch_session_messages(session_id)


def refresh_history():
    """ Retrieve and update the global history metadata list."""
    try:
        history_resp = requests.get(f"{BACKEND_URL}/sessions/get_list", timeout=settings.INTERNAL_TIMEOUT)
        st.session_state.history_list = history_resp.json() if history_resp.status_code == 200 else []
    except requests.exceptions.RequestException:
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
            if st.button(f"💬 {hist['title']}", key=f"btn_{hist['session_id']}",
                         type="primary" if is_active else "secondary", use_container_width=True):
                load_session(hist['session_id'], hist['title'])
                st.rerun()

        with col2:
            if st.button("🗑️", key=f"del_{hist['session_id']}"):
                requests.delete(f"{BACKEND_URL}/sessions/{hist['session_id']}", timeout=settings.INTERNAL_TIMEOUT)
                if hist["session_id"] == st.session_state.session_id:
                    del st.session_state.app_initialized
                    st.rerun()


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
