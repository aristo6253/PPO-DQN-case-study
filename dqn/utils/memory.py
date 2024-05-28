import collections
from collections import namedtuple, deque
import random

class ExpMem:
    """
    This class implements the replay memory as mentionned in the paper.

    The constructor just sets the set its size and the number of initial transitions and implements the memory itself as a deque, a convenient data structure that automatically pops the last elements if the number of elements in the structure is bigger than the maximum value set
    The sample random method allows us to randomly sample a batch of transitions
    The append method allows us to add transitions to the memory
    """
    def __init__(self, mem_size=10000, new_transitions=3000):
        self.mem_size = mem_size
        self.new_trans = new_transitions
        self.mem = collections.deque([], maxlen=mem_size)
    def sample_random(self, batch_size=32):
        return random.sample(self.mem, batch_size)
    def append(self, transition):
        self.mem.append(transition) 

