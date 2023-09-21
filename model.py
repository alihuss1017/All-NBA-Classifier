import torch.nn as nn
import numpy as np
import pandas as pd

df = pd.read_csv('DataSet.csv')
df = df.loc[:,['GS', 'FG', 'FGA', '2P', '2PA', 'AST', 'TOV', 'PTS', 'WS', 'VORP']]

class CustomModel(nn.Module):
    def __init__(self, num_layers, num_neurons, input_size = len(df.columns)):
        super(CustomModel, self).__init__()
        self.layers = [nn.Linear(input_size, num_neurons), nn.ReLU()]

        for layer in range(1, num_layers):
            self.layers.append(nn.Linear(num_neurons, num_neurons))
            self.layers.append(nn.ReLU())
        
        self.layers.append(nn.Linear(num_neurons, 1))
        self.layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)
    
    def feature_scaling(self, x):
        scaled_feats = []
        for n in range(len(df.columns)):
            scaled_feats.append((x[n] - np.mean(df.iloc[:,n])) / np.std(df.iloc[:,n]))  
        return scaled_feats