import torch.nn as nn
import numpy as np
import pandas as pd

df = pd.read_csv('Dataset.csv')
df = df.loc[:,['GS', 'FG', 'FGA', '2P', '2PA', 'AST', 'TOV', 'PTS', 'WS', 'VORP']]

class CustomModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(CustomModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
    
    def feature_scaling(self, x):
        scaled_feats = []
        for n in range(len(df.columns)):
            scaled_feats.append((x[n] - np.mean(df.iloc[:,n])) / np.std(df.iloc[:,n]))  
        return scaled_feats