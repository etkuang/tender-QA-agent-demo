# coding: utf-8
# @Author: Wang Qingkang

import uuid

import streamlit as st

from app_frontend_layer.components.chat_view import render_chat
from app_frontend_layer.components.sidebar import load_session, refresh_history, render_sidebar
from app_frontend_layer.config import load_frontend_config


def init_state():
    """Initializes global frontend state execution context on application startup."""
    if "app_initialized" not in st.session_state:
        config_data = load_frontend_config()

        st.session_state.temp = config_data.get("temp", 0.7)
        st.session_state.top_p = config_data.get("top_p", 0.95)
        st.session_state.max_t = config_data.get("max_t", 2048)

        refresh_history()

        if st.session_state.history_list:
            first_session = st.session_state.history_list[0]
            load_session(first_session["session_id"], first_session["title"])
        else:
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.session_title = "New Chat"
            st.session_state.messages = []

        st.session_state.app_initialized = True


def main():
    """Application entry point."""
    st.set_page_config(page_title="Bidding Assistant", layout="wide")
    init_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()