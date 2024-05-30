import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from typing import List, Tuple
import cv2

import collections
from collections import namedtuple, deque
import tqdm
import matplotlib.pyplot as plt
import random
import gymnasium as gym
import pandas as pd

from dqn.models.Q_agent import Architecture

torch.manual_seed(0)
np.random.seed(0)

def train_loop(environment):
    env_name = environment
    num_episodes_train = 1500 
    num_episodes_test = 20
    learning_rate = 5e-4

    env = gym.make(env_name)
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.shape[0]

    num_seeds = 3 
    l = num_episodes_train // 10
    res = np.zeros((num_seeds, l))
    all_rewards = []
    gamma = 0.99
    compteur = 0

    for i in tqdm.tqdm(range(num_seeds)):
        reward_means = []
        rewards = []
        seed = i + 1  
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        agent = Architecture(env_name, lr=learning_rate, actions=action_space_size, states=state_space_size)

        for m in range(num_episodes_train):
            agent.train()

            if m % 10 == 0:
                print("Episode: {}".format(m))

                G = np.zeros(num_episodes_test)
                for k in range(num_episodes_test):
                    g = agent.test()
                    G[k] = g

                reward_mean = G.mean()
                reward_sd = G.std()
                print(f"The test reward for episode {m} is {reward_mean} with a standard deviation of {reward_sd}.")
                reward_means.append(reward_mean)

            rewards.append(agent.test())
            compteur += 1

        res[i] = np.array(reward_means)
        all_rewards.append(rewards)

    torch.save(agent.learning_net.net.state_dict(), f'./weights/trained_model_{env_name}_{learning_rate}_smallbuffer.pth')
    print("Model weights saved to trained_model.pth")

    df_dqn = pd.DataFrame(all_rewards).transpose()
    df_dqn.columns = [f'seed_{i+1}' for i in range(num_seeds)]
    df_dqn['episode'] = df_dqn.index

    df_dqn.to_csv('./csv/dqn_rewards.csv', index=False)
    print("DQN rewards saved to dqn_rewards.csv")
    
    ks = np.arange(l) * 10
    avs = np.mean(res, axis=0)
    maxs = np.max(res, axis=0)
    mins = np.min(res, axis=0)

    plt.fill_between(ks, mins, maxs, alpha=0.1)
    plt.plot(ks, avs, '-o', markersize=1)

    plt.xlabel('Episode', fontsize=15)
    plt.ylabel('Avg. Return', fontsize=15)
    plt.show()

    return agent

def visual(agent, environment):
    env = gym.make(environment, render_mode='rgb_array')
    state, _ = env.reset()
    cv2.namedWindow(environment, cv2.WINDOW_NORMAL)  

    for i in range(200):
        frame = env.render()
        frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        cv2.imshow(environment, frame_bgr) 

        if cv2.waitKey(25) & 0xFF == ord('q'): 
            break
        
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            q_values = agent.learning_net.net(state)
        action = agent.e_greedy(q_values).detach().numpy().item()
        
        state, reward, terminated, truncated, _ = env.step(action)
        
        if terminated or truncated:
            print(f"terminated status: {terminated} \ntruncation status:{truncated}")
            state, _ = env.reset()

    cv2.destroyAllWindows()
    env.close()


def main():
    env = "CartPole-v1"
    agent = train_loop(env)
    visual(agent, env)


if __name__ == '__main__':
    main()
