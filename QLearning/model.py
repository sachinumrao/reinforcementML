import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.fc_1 = nn.Linear()
        self.fc_2 = nn.Linear()
        
    def forward(self, x):
        out = self.fc_1(x)
        out = F.relu(out)
        out = self.fc_2(out)
        return out
        