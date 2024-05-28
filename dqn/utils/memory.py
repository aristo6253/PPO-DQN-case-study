import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from typing import List, Tuple

import collections
from collections import namedtuple, deque
import random

from IPython.display import clear_output
from IPython import display

class ExpMem:
    def __init__(self, mem_size=10000, new_transitions=3000):
        # Constructor of the Experience Memory which stores transitions 10'000 / 100'000 at a time
        self.mem_size = mem_size
        self.new_trans = new_transitions
        self.mem = collections.deque([], maxlen=mem_size)
    def sample_random(self, batch_size=32):
        # Samples a random batch of transitions to train the model
        return random.sample(self.mem, batch_size)
    def append(self, transition):
        # Adds a transition to the right of the memory
        self.mem.append(transition) 

