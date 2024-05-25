import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gym
import time
from scipy.stats import ttest_ind


def load_data_ours(dir_ours, key='', random=False):
    avg_ep_ret = []
    avg_ep_len = []
    critic_loss = []
    actor_loss = []
    timesteps = []
    data = {}

    for dir in os.listdir(dir_ours):
        if key in dir:
          aer, ael, cl, al, t = extract_our_data(dir_ours + dir, random=random)
          avg_ep_ret.append(aer)
          avg_ep_len.append(ael)
          critic_loss.append(cl)
          actor_loss.append(al)
          timesteps.append(t)
    
    data['avg_ep_ret'] = pd.DataFrame(avg_ep_ret)
    data['avg_ep_len'] = pd.DataFrame(avg_ep_len)
    data['critic_loss'] = pd.DataFrame(critic_loss)
    data['actor_loss'] = pd.DataFrame(actor_loss)
    data['timesteps'] = pd.DataFrame(timesteps)

    return data

def load_data_baseline(dir_base):
    ep_len_mean = []
    ep_rew_mean = []
    timesteps = []
    data = {}

    for dir in os.listdir(dir_base):
        mean_rew, mean_len, t = extract_baseline_data(dir_base + dir)
        ep_rew_mean.append(mean_rew)
        ep_len_mean.append(mean_len)
        timesteps.append(t)
    
    data['avg_ep_ret'] = pd.DataFrame(ep_rew_mean)
    data['avg_ep_len'] = pd.DataFrame(ep_len_mean)
    data['timesteps'] = pd.DataFrame(timesteps)

    return data


def extract_our_data(filename, random=False):
    avg_ep_ret = []
    avg_ep_len = []
    critic_loss = []
    actor_loss = []
    timesteps = []

    with open(filename, 'r') as file:
      data = file.read()
      data = '[' + data.replace('}\n{', '},{') + ']'

      json_data = json.loads(data)

      for entry in json_data:
          avg_ep_ret.append(entry['avg_ep_ret'])
          avg_ep_len.append(entry['avg_ep_len'])
          if not random:
              actor_loss.append(entry['actor_loss'])
              critic_loss.append(entry['critic_loss'])
          timesteps.append(entry['timesteps'])

      return avg_ep_ret, avg_ep_len, critic_loss, actor_loss, timesteps

def add_plot_seed_avg(data_x, data_y, label=None, color='blue', metric='mean'):
  
  if metric == 'mean':
    means = data_x.mean(axis=0)
    sem = data_x.sem(axis=0)
    plt.plot(data_y, means, label=label, color=color)
    plt.fill_between(data_y, means - 1.96 * sem, means + 1.96 * sem, alpha=0.2, color=color)
  if metric == 'median':
    median = data_x.median(axis=0)
    p_10 = data_x.quantile(0.1)
    p_90 = data_x.quantile(0.9)
    plt.plot(data_y, median, label=label, color=color)
    plt.fill_between(data_y, p_10, p_90, alpha=0.2, color=color)


  

def extract_baseline_data(filename):
  x1, x2, x3 = [], [], []
  with open(filename, 'r') as f:
    for l in f:
        l = [e.strip() for e in l.split('|')]
        if 'ep_rew_mean' in l:
            x1.append(float(l[2]))
        if 'ep_len_mean' in l:
            x2.append(float(l[2]))
        if 'total_timesteps' in l:
            x3.append(int(l[2]))
  return x1, x2, x3

def run_random_policy(env_name, num_timesteps, batch_size, seed, name):
    env = gym.make(env_name)
    env.seed(seed)
    np.random.seed(seed)

    state = env.reset()

    total_reward = 0
    episode_rewards = []
    episode_lengths = []
    timesteps = 0

    batch_rewards = []
    batch_lengths = []

    logger = {
        "dt": time.time_ns(),
        "timesteps": 0,
        "iterations": 0,
        "episode_lengths": [],
        "episode_returns": [],
    }

    data = []

    while timesteps < num_timesteps:
        state = env.reset()
        done = False
        ep_reward = 0
        ep_length = 0

        while not done:
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)

            ep_reward += reward
            ep_length += 1
            timesteps += 1

            if timesteps % batch_size == 0:
                avg_ep_len = np.mean(logger["episode_lengths"]) if logger["episode_lengths"] else 0
                avg_ep_ret = np.mean([np.sum(ep) for ep in logger["episode_returns"]]) if logger["episode_returns"] else 0
                batch_result = {
                    "avg_ep_ret": avg_ep_ret,
                    "avg_ep_len": avg_ep_len,
                    "timesteps": timesteps
                }
                data.append(batch_result)
                logger["episode_lengths"] = []
                logger["episode_returns"] = []

        logger["episode_lengths"].append(ep_length)
        logger["episode_returns"].append(ep_reward)

    env.close()


    with open(f'./random_seed_data/{name}/data_{seed}.json', 'w') as outfile:
        for i in data:
            json_string = json.dumps(i, indent=4, default=float)
            outfile.write(json_string + '\n')

def calculate_ttest(df1, df2):
    mean1 = df1.mean(axis=1)
    mean2 = df2.mean(axis=1)

    t_stat, p_value = ttest_ind(mean1, mean2)

    print("T-statistic:", t_stat)
    print("P-value:", p_value)



