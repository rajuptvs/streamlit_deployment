import streamlit as st
from multiapp import MultiApp
from apps import rating_model,tagging_model



st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)
st.sidebar.success("Select a demo above.")
# Add all your application here
