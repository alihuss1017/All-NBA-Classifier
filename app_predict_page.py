import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import path

from model import CustomModel

dir = path.Path(__file__).abspath()
sys.path.append(dir.parent.parent)

def predict_page():

    st.title('NBA All-Team Binary Classification Model')

    min_values = np.load('min_init_values.npy')
    max_values = np.load('max_init_values.npy')

    loaded_data = torch.load('nbamodel.pt')

    # Extract the hyperparameters
    hyperparams = loaded_data['best_hyperparameters']
    num_neurons, num_layers = hyperparams['module__num_neurons'], hyperparams['module__num_layers']
    # Create an instance of your CustomModel using the loaded hyperparameters
    model = CustomModel(num_layers, num_neurons)

    # Load the model's state_dict
    model.load_state_dict(loaded_data['model_state_dict'])

    GS = st.text_input('Games Started', min_values[0], max_values[0])
    FG = st.text_input('Field Goals Made', min_values[1], max_values[1])
    FGA = st.text_input('Field Goals Attempted', min_values[2], max_values[2])
    two_p = st.text_input('Two Pointers Made', min_values[3], max_values[3])
    two_pa = st.text_input('Two Pointers Attempted', min_values[4], max_values[4])
    AST = st.text_input('Assists', min_values[5], max_values[5])
    TOV = st.text_input('Turnovers', min_values[6], max_values[6])
    PTS = st.text_input('Points', min_values[7], max_values[7])
    WS = st.text_input('Win Shares', min_values[8], max_values[8])
    VORP = st.text_input('Value Over Replacement', min_values[9], max_values[9])

    features = [GS, FG, FGA, two_p, two_pa, AST, TOV, PTS, WS, VORP]
    features = model.feature_scaling(features)

    input_data = torch.tensor(features,  
                            dtype=torch.float32)

    model.eval()

    with torch.no_grad():
        predictions = model(input_data)
        predictions = (predictions > 0.8).float()
    st.write('Predictions:', predictions.numpy())