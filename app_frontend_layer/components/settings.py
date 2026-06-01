# coding: utf-8
# @Author: Wang Qingkang

import streamlit as st

from app_frontend_layer.config import save_frontend_config


@st.dialog("LLM Configuration")
def render_settings_dialog():
    """Render the configuration management UI inside a modal dialog window."""
    col1, col2, col3 = st.columns(3)
    with col1:
        new_temp = st.slider("Temperature", min_value=0.0, max_value=2.0, value=st.session_state.temp, step=0.1)
    with col2:
        new_top_p = st.slider("Top-P", min_value=0.0, max_value=1.0, value=st.session_state.top_p, step=0.05)
    with col3:
        new_max_t = st.number_input(
            "Max Tokens",
            min_value=256,
            max_value=32768,
            value=st.session_state.max_t,
            step=256,
        )

    st.divider()

    col_spacer, col_cancel, col_submit = st.columns([3, 1, 1])
    with col_cancel:
        if st.button("Cancel", use_container_width=True):
            st.rerun()
    with col_submit:
        if st.button("Save", type="primary", use_container_width=True):
            payload = {
                "temp": new_temp,
                "top_p": new_top_p,
                "max_t": new_max_t,
            }
            save_frontend_config(payload)

            st.session_state.temp = new_temp
            st.session_state.top_p = new_top_p
            st.session_state.max_t = new_max_t
            st.rerun()
