import collections
from collections import namedtuple, deque
import random

class ExpMem:
    def __init__(self, mem_size=10000, new_transitions=3000):
        self.mem_size = mem_size
        self.new_trans = new_transitions
        self.mem = collections.deque([], maxlen=mem_size)
    def sample_random(self, batch_size=32):
        return random.sample(self.mem, batch_size)
    def append(self, transition):
        self.mem.append(transition) 

