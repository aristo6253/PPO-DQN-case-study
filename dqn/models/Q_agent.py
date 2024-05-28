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

from .base_architecture import Q_model
from utils.memory import ExpMem

Recording = namedtuple('Recording', ('state', 'action', 'next_state', 'reward'))

#actions = 2
#states = 4

class Architecture:
    def __init__(self, env_name, lr=5e-2, render=False, actions=2, states=4, mem_size=10000): #changes mem_size to 1000 
        # Class constructor
        #print(f"mem size is {mem_size = }")
        self.env = gym.make(env_name)
        self.lr = lr
        self.learning_net = Q_model(self.env, self.lr, actions=actions, states=states)
        self.target_net = Q_model(self.env, self.lr, actions=actions, states=states)
        self.target_net.net.load_state_dict(self.learning_net.net.state_dict())
        self.mem = ExpMem(mem_size=mem_size)

        self.save_in_mem()
        self.batch_size = 32
        self.gamma = 0.99
        self.c = 0
        
    def save_in_mem(self):
        # We have to initialize the experience memory with a number of episodes
        cnt = 0
        terminated = False
        truncated = False
        state, _ = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        while cnt < self.mem.new_trans:
            if terminated or truncated:
                state, _ = self.env.reset()
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            action = self.env.action_space.sample()

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            reward = torch.tensor([reward])
            if terminated: # Because it still returns a non-empty next_state 
                next_state = None
            else:
                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                #print(f"{next_state.shape = }")
            
            # Store new experience into memory
            transition = Recording(state, action, next_state, reward)
            self.mem.mem.append(transition)
            state = next_state
            cnt += 1
        
    def e_greedy(self, q_vals, epsilon=0.05):
        #print(f"{q_vals = }")
        p = random.random()
        if p > epsilon:
            with torch.no_grad():
                return torch.argmax(q_vals).unsqueeze(0).unsqueeze(0)
        else:
            return torch.tensor([[self.env.action_space.sample()]], dtype=torch.long)
    
    def train(self):
        # Train the Q-network using Deep Q-learning.
        state, _ = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        terminated = False
        truncated = False

        # Loop until reaching the termination state
        while not (terminated or truncated):
            with torch.no_grad():
                q_values = self.learning_net.net(state)
            #print(f"{q_values = }")
            action = self.e_greedy(q_values).reshape(1,1).item()
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            reward = torch.tensor([reward])
            
            if terminated:
                next_state = None
            else: 
                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            
            # Store the new experience
            transition = Recording(state, action, next_state, reward)
            self.mem.mem.append(transition)
            state = next_state

            transitions = self.mem.sample_random(self.batch_size)
            batch = Recording(*zip(*transitions))
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
            #non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
            non_final_next_states = [s for s in batch.next_state if s is not None]
            if len(non_final_next_states) == 0:
                return  # Skip this batch or handle differently

            non_final_next_states = torch.cat(non_final_next_states)

            state_batch = torch.cat(batch.state)
            action_batch = torch.tensor(batch.action, dtype=torch.int64).unsqueeze(1)  # Convert to tensor and unsqueeze
            reward_batch = torch.cat(batch.reward)

            # Get current and next state values
            #print(f"{state_batch.shape = } \n{action_batch.shape = }")
            state_action_values = self.learning_net.net(state_batch).gather(1, action_batch) # extract values corresponding to the actions Q(S_t, A_t)
            next_state_values = torch.zeros(self.batch_size)
            
            with torch.no_grad():
                # no next_state_value update if an episode is terminated (next_satate = None)
                # only update the non-termination state values (Ref: https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)
                next_state_values[non_final_mask] = self.target_net.net(non_final_next_states).max(1)[0] # extract max value
                
            # Update the model
            expected_state_action_values = (next_state_values * self.gamma) + reward_batch
            criterion = torch.nn.MSELoss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
            self.learning_net.optimizer.zero_grad()
            loss.backward()
            self.learning_net.optimizer.step()

            # Update the target Q-network in each 50 steps
            self.c += 1
            if self.c % 50 == 0:
                self.target_net.net.load_state_dict(self.learning_net.net.state_dict())
            

    def test(self, model_file=None):
        # Evaluates the performance of the agent over 20 episodes.

        max_t = 1000
        state, _ = self.env.reset()
        rewards = []

        for t in range(max_t):
            state = torch.from_numpy(state).float().unsqueeze(0)
            with torch.no_grad():
                q_values = self.learning_net.net(state)
            action = torch.argmax(q_values).item() #self.e_greedy(q_values)
            state, reward, terminated, truncated, _ = self.env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                break

        return np.sum(rewards)