# coding: utf-8
# @Author: Wang Qingkang

import streamlit as st

from app_frontend_layer.components.chat_view import render_chat
from app_frontend_layer.components.sidebar import load_session, refresh_history, render_sidebar


def init_state():
    """Initializes global frontend state execution context on application startup."""
    if "app_initialized" not in st.session_state:

        refresh_history()

        if st.session_state.history_list:
            first_session = st.session_state.history_list[0]
            load_session(first_session["session_id"], first_session["title"])
        else:
            st.session_state.session_id = None
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