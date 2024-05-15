import streamlit as st

st.set_page_config(page_title="Activity")

st.title('Activity')

with st.container(border=True):
    st.write(st.session_state.activity)

