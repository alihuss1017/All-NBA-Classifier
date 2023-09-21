import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import path
from app_about_page import about_page
from app_predict_page import predict_page

dir = path.Path(__file__).abspath()
sys.path.append(dir.parent.parent)

st.set_page_config("Ali's NBA Web App")

page = st.sidebar.selectbox("Navigate", ["About", "Predict"])

if page == "About":
    about_page()

if page == "Predict":
    predict_page()

