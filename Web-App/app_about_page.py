import streamlit as st

def about_page():
    with open("README.md", "r") as md_file:
        md_text = md_file.read()

    st.markdown(md_text)