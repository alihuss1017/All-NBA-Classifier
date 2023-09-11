import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import path

dir = path.Path(__file__).abspath()
sys.path.append(dir.parent.parent)


from model import CustomModel
st.set_page_config("Ali's NBA Web App")
st.title('NBA All-Team Binary Classification Model')

min_values = np.load('min_init_values.npy')
max_values = np.load('max_init_values.npy')

model = CustomModel(input_size = 10, hidden_size1 = 32, hidden_size2 = 16) 
model.load_state_dict(torch.load('nbamodel.pt', map_location=torch.device('cpu')))


GS = st.slider('Games Started', min_values[0], max_values[0])
FG = st.slider('Field Goals Made', min_values[1], max_values[1])
FGA = st.slider('Field Goals Attempted', min_values[2], max_values[2])
two_p = st.slider('Two Pointers Made', min_values[3], max_values[3])
two_pa = st.slider('Two Pointers Attempted', min_values[4], max_values[4])
AST = st.slider('Assists', min_values[5], max_values[5])
TOV = st.slider('Turnovers', min_values[6], max_values[6])
PTS = st.slider('Points', min_values[7], max_values[7])
WS = st.slider('Win Shares', min_values[8], max_values[8])
VORP = st.slider('Value Over Replacement', min_values[9], max_values[9])

features = [GS, FG, FGA, two_p, two_pa, AST, TOV, PTS, WS, VORP]
features = model.feature_scaling(features)

input_data = torch.tensor(features,  
                          dtype=torch.float32)

model.eval()

with torch.no_grad():
    predictions = model(input_data)

st.write('Predictions:', predictions.numpy())
