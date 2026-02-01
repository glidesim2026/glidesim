import streamlit as st

from glidesim.ui.results import render_results
from glidesim.ui.sidebar import render_sidebar

st.set_page_config(page_title="GlideSim", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        min-width: 450px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("GlideSim")

render_sidebar()

render_results(
    st.session_state.results,
    st.session_state.metrics,
    st.session_state.config,
)
