import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from typing import List, Tuple

import collections
from collections import namedtuple, deque
import tqdm
import matplotlib.pyplot as plt
import random
import gymnasium as gym

from IPython.display import clear_output
from IPython import display


class NeuralNetwork(nn.Module):
    def __init__(self, in_size, out_size):
        # Init the network 
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(in_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 128)
        self.outLayer = nn.Linear(128, out_size)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

        # Xavier init the weights
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear3.weight)
        nn.init.xavier_uniform_(self.outLayer.weight)
    
    def forward(self, inputs):
        # Forward pass of the network
        out = self.relu1(self.linear1(inputs))
        out = self.relu2(self.linear2(out))
        out = self.relu3(self.linear3(out))
        out = self.outLayer(out)
        return out


class Q_model:
    def __init__(self, env, lr, states, actions, logdir=None):
        # Define Q-network with specified architecture
        self.net = NeuralNetwork(states, actions)
        self.env = env
        self.lr = lr 
        self.logdir = logdir
        self.optimizer = optim.AdamW(self.net.parameters(), lr=self.lr)

    def load_model(self, model_file):
        # Load pre-trained model from a file
        return self.net.load_state_dict(torch.load(model_file))

