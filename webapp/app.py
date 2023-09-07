import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import path

dir = path.Path(__file__).abspath()
sys.path.append(dir.parent.parent)

from models.model import CustomModel 

st.title('NBA All-Team Binary Classification Model')


min_values = np.load('Data/min_values.npy')
max_values = np.load('Data/max_values.npy')

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

input_data = torch.tensor([GS, FG, FGA, two_p, two_pa, AST, TOV, PTS, WS, VORP], 
                          dtype=torch.float32)

model = CustomModel(input_size = 10, hidden_size1 = 32, hidden_size2 = 16) 
model.load_state_dict(torch.load('models/nbamodel.pt', map_location=torch.device('cpu')))

model.eval()

with torch.no_grad():
    predictions = model(input_data)

st.write('Predictions:', predictions.numpy())
