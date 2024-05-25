import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


# This network will define the Actor and Critic networks
# Input: State
# Output: Action (Actor), Value (Critic)
class FFNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FFNN, self).__init__()
        # The hidden dimension was arbitrarily chosen to be 64 (doesn't really matter)
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, out_dim)

    def forward(self, x, device='cpu'):
        # print(f"{x = }")
        # ReLU was again chosen arbitrarily
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float).to(device)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

