from ppo.network import FFNN
import torch
from torch import nn
from torch.distributions import MultivariateNormal
import time
import numpy as np
import matplotlib.pyplot as plt
import csv, json

class PPO_ours:
    def __init__(self, env, seed=42, gamma=0.99, batch_size=4800, max_timesteps=1e3, 
                 updates_per_iter=5, clip=0.2, lr=5e-3, entropy_reg=False, lr_annealing=False,
                 adv_norm=False, grad_clip=False, render=False, render_freq=10, save=True, 
                 save_freq=10, ipynb=False, name='', verbose=None, device='cpu'):
        
        self.device = device
        self.env = env
        self.seed = seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        env.seed(self.seed)
        
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.batch_size = batch_size
        self.max_timesteps= max_timesteps
        self.updates_per_iter = updates_per_iter
        self.gamma = gamma
        self.clip = clip
        self.lr = lr

        self.adv_norm = adv_norm
        self.entropy_reg = entropy_reg
        self.ent_coef = 0.01 if entropy_reg else 0
        self.lr_annealing = lr_annealing
        self.grad_clip = grad_clip

        self.name = name
        self.render = render
        self.render_freq = render_freq
        self.save = save
        self.save_freq = save_freq
        self.ipynb = ipynb
        self.verbose = verbose

        self.actor = FFNN(self.state_dim, self.action_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic = FFNN(self.state_dim, 1).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        
        self.variance = torch.full((self.action_dim,), 0.5).to(self.device)
        self.covariance = torch.diag(self.variance).to(self.device)


        self.avg_ep_len = []
        self.avg_ep_ret = []
        self.actor_loss = []
        self.critic_loss = []
        self.timesteps = []
        self.data = []

        self.logger = {
            "dt": time.time_ns(),
            "timesteps": 0,
            "iterations": 0,
            "episode_lengths": [],
            "episode_returns": [],
            "actor_loss": [],
            "critic_loss": []
        }
        if verbose == 'heavy':
          print(f"(seed = {self.seed})")
          print(f"Working with environment {self.name}, which has {self.state_dim} states and {self.action_dim} actions.")

    # The learning part
    def learn(self, T):
        self.expected_iters = np.ceil(T/self.batch_size)
        if self.verbose == 'light':
          print("*"*50)
          print(f"Starting training for {self.name} (seed = {self.seed})...")
          print(f"Running PPO for {T} timesteps...")
          print(f"Batch size: {self.batch_size}")
          print(f"Max timesteps per episode: {self.max_timesteps}")
          print(f"Expected Iterations: {self.expected_iters}")
          print("*"*50)
        t = 0   # Current time step
        i = 0   # Current iteration

        while t < T:
            states, actions, log_probs, r2g, ep_lengths = self.rollout()

            t += sum(ep_lengths)
            i += 1

            self.logger["timesteps"] = t
            self.logger["iterations"] = i

            V, _, _ = self.evaluate(states, actions)
            A = r2g - V.detach()
            if self.adv_norm:
              A = (A - A.mean()) / (A.std() + 1e-10)

            for _ in range(self.updates_per_iter):
                if self.lr_annealing:
                    frac = (t - 1.0) / T
                    new_lr = self.lr * (1.0 - frac)
                    new_lr = max(new_lr, 0.0)
                    self.actor_optimizer.param_groups[0]["lr"] = new_lr
                    self.critic_optimizer.param_groups[0]["lr"] = new_lr

                V, log_probs_curr, entropy = self.evaluate(states, actions)

                ratios = torch.exp(log_probs_curr - log_probs)

                surr1 = ratios * A
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A

                actor_loss = (-torch.min(surr1, surr2)).mean()
                entropy_loss = entropy.mean()
                actor_loss = actor_loss - self.ent_coef * entropy_loss

                critic_loss = nn.MSELoss()(V, r2g)

                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
                self.critic_optimizer.step()

                self.logger["actor_loss"].append(actor_loss.detach())
                self.logger["critic_loss"].append(critic_loss.detach())
            
            self.print_logs()

            if self.save and i % self.save_freq == 0:
                if self.ipynb:
                    torch.save(self.actor.state_dict(), f"./ppo/checkpoints/actor/{self.name}/actor_{i}.pth")
                    torch.save(self.critic.state_dict(), f"./ppo/checkpoints/critic/{self.name}/critic_{i}.pth")
                    self.save_plots(path=f"./ppo/plots/{self.name}/")
                else:
                    torch.save(self.actor.state_dict(), f"./Project/ppo/checkpoints/actor/{self.name}/actor_{i}.pth")
                    torch.save(self.critic.state_dict(), f"./Project/ppo/checkpoints/critic/{self.name}/critic_{i}.pth")
                    self.save_plots(path=f"./Project/plots/{self.name}/")
                self.save_plots(path=f"./ppo/plots/{self.name}/", final=True)
        self.save_seed_results() 
  
    def rollout(self):

        states = []
        actions = []
        log_probs = []
        rewards = []
        r2g = []
        ep_lengths = []
        ep_rewards = []

        t = 0

        while t < self.batch_size:
            ep_rewards = []
            if self.ipynb:
                state = self.env.reset()
            else:
                state, _ = self.env.reset()
            done = False

            for i in range(int(self.max_timesteps)):
                if self.render and self.logger["iterations"] % self.render_freq == 0:
                    self.env.render()
                
                t += 1  # Increment time step for current batch

                states.append(state)

                action, log_prob = self.get_action(state)

                # Get response from the environment
                if self.ipynb:
                    state, reward, done, _ = self.env.step(action)
                else:
                    state, reward, done, _, _ = self.env.step(action)

                # Record the action, log_prob, and reward
                actions.append(action)
                log_probs.append(log_prob)
                ep_rewards.append(reward)


                if done:
                    break

            ep_lengths.append(i + 1)
            rewards.append(ep_rewards)

        # Convert lists to tensors
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.device)
        log_probs = torch.tensor(log_probs, dtype=torch.float).to(self.device)
        r2g = self.compute_r2g(rewards)

        self.logger['episode_lengths'] = ep_lengths
        self.logger['episode_returns'] = rewards

        return states, actions, log_probs, r2g, ep_lengths

    def get_action(self, state):
        mean = self.actor(state, device=self.device)
        dist = MultivariateNormal(mean, self.covariance)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().cpu().numpy(), log_prob.detach()
    
    def compute_r2g(self, rewards):
        r2g = []

        for er in reversed(rewards):
            cum_reward = 0
            for r in reversed(er):
                cum_reward = r + self.gamma * cum_reward
                r2g.insert(0, cum_reward)
        return torch.tensor(r2g, dtype=torch.float).to(self.device)
    
    def evaluate(self, states, actions):
        V = self.critic(states).squeeze()
        mean = self.actor(states)
        dist = MultivariateNormal(mean, self.covariance)
        log_probs = dist.log_prob(actions)
        return V, log_probs, dist.entropy()

    def print_logs(self):
        dt = self.logger["dt"]
        self.logger["dt"] = time.time_ns()
        dt = str(round((self.logger["dt"] - dt) / 1e9, 3))

        timesteps = self.logger["timesteps"]
        iterations = self.logger["iterations"]

        avg_ep_len = np.mean(self.logger["episode_lengths"])
        avg_ep_ret = np.mean([np.sum(i) for i in self.logger["episode_returns"]] )
        
        avg_actor_loss = np.mean([losses.detach().cpu().numpy().mean() for losses in self.logger['actor_loss']])
        avg_critic_loss = np.mean([losses.detach().cpu().numpy().mean() for losses in self.logger['actor_loss']])

        avg_ep_len_str = str(round(avg_ep_len, 3))
        avg_ep_ret_str = str(round(avg_ep_ret, 3))
        avg_actor_loss_str = str(round(avg_actor_loss, 5))
        avg_critic_loss_str = str(round(avg_critic_loss, 5))

        if self.verbose == 'heavy':
            print(flush=True)
            print(f"-------------------- Iteration #{iterations:,} --------------------", flush=True)
            print(f"Time elapsed: {dt} seconds", flush=True)
            print(f"Timesteps: {timesteps:,}", flush=True)
            print(f"Average Episodic Length: {avg_ep_len_str}", flush=True)
            print(f"Average Episodic Return: {avg_ep_ret_str}", flush=True)
            print(f"Average Actor Loss: {avg_actor_loss_str}", flush=True)
            print(f"Average Critic Loss: {avg_critic_loss_str}", flush=True)
            print(f"------------------------------------------------------", flush=True)
            print(f'{self.logger["episode_returns"].shape = }')
        if self.verbose == 'light' and iterations % 10 == 0:
            print(f"Iteration {iterations:,}/{self.expected_iters:.0f}")

        

        self.avg_ep_len.append(avg_ep_len)
        self.avg_ep_ret.append(avg_ep_ret)
        self.actor_loss.append(avg_actor_loss)
        self.critic_loss.append(avg_critic_loss)
        self.timesteps.append(timesteps)
        self.data.append({'avg_ep_ret': avg_ep_ret, 'avg_ep_len': avg_ep_len, 'actor_loss': avg_actor_loss, 'critic_loss': avg_critic_loss, 'timesteps': timesteps})

    def plot(self, all=True, avg_ep_len=True, avg_ep_ret=True, actor_loss=True, critic_loss=True):

        if all:
            avg_ep_len = True
            avg_ep_ret = True
            actor_loss = True
            critic_loss = True

        if avg_ep_len:
            plt.plot(self.avg_ep_len)
            plt.title("Average Episodic Length")
            plt.xlabel("Iterations")
            plt.ylabel("Length")
            plt.show()

        if avg_ep_ret:
            plt.plot(self.avg_ep_ret)
            plt.title("Average Episodic Return")
            plt.xlabel("Iterations")
            plt.ylabel("Return")
            plt.show()

        if actor_loss:
            plt.plot(self.actor_loss)
            plt.title("Average Actor Loss")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.show()

        if critic_loss:
            plt.plot(self.critic_loss)
            plt.title("Average Critic Loss")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.show()

    def save_plots(self, path="./", final=False):

        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle("PPO Training Metrics")
        fig.tight_layout()

        axs[0].plot(self.avg_ep_ret)
        axs[0].set_title("Average Episodic Return")
        axs[0].set_xlabel("Iterations")
        axs[0].set_ylabel("Return")

        axs[1].plot(self.actor_loss)
        axs[1].set_title("Average Actor Loss")
        axs[1].set_xlabel("Iterations")
        axs[1].set_ylabel("Loss")

        axs[2].plot(self.critic_loss)
        axs[2].set_title("Average Critic Loss")
        axs[2].set_xlabel("Iterations")
        axs[2].set_ylabel("Loss")

        axs[3].plot(self.avg_ep_len)
        axs[3].set_title("Average Episodic Length")
        axs[3].set_xlabel("Iterations")
        axs[3].set_ylabel("Length")
        if final:
            plt.savefig(path + f"{self.seed}_{self.logger['iterations']}_final.png")
        else:
            plt.savefig(path + f"{self.seed}_{self.logger['iterations']}.png")
        plt.close(fig)

    def save_seed_results(self):

        name = f'data_{self.seed}'
        if self.adv_norm and self.lr_annealing and self.grad_clip:
          name += '_full'
        else:
          if self.adv_norm:
            name += '_n'
          if self.lr_annealing:
            name += '_lr'
          if self.grad_clip:
            name += '_gc'
        
        if self.adv_norm or self.lr_annealing or self.grad_clip:
            if sum([self.adv_norm, self.entropy_reg, self.lr_annealing, self.grad_clip]) == 1:
              name += '_alone'
        else:
          name += '_basic'
        
        name +='.json'

        with open(f'./ppo/data/seed_data/{self.name}/' + name, 'w') as outfile:
            for i in self.data:
                json_string = json.dumps(i, indent=4, default=float)
                outfile.write(json_string + '\n')

        