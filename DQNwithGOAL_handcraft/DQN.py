import gym
from my_env import BitFlip
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import collections
import random
import torch.nn.functional as F
import ipdb
import matplotlib.pyplot as plt
# import rl_utils
import hydra
import os
import wandb
import omegaconf

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done, goal):
        self.buffer.append((state, action, reward, next_state, done, goal))

    def sample(self, batch_size):
        transition = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, goal = zip(*transition)
        return np.array(state), action, reward, np.array(next_state), done, goal

    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, name, checkpoint_dir):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = os.path.join(checkpoint_dir, name) + ".pth"

    def forward(self, x):
        # ipdb.set_trace()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def save_checkpoint(self):
        print(f"Saving checkpoint at {self.checkpoint_name}")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        torch.save(self.state_dict(), self.checkpoint_name)
    
    def load_checkpoint(self):
        print(f"Loading checkpoint at {self.checkpoint_name}")
        self.load_state_dict(torch.load(self.checkpoint_name))

def save_model(model, episode, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, f"model_episode_{episode}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")


class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device, length, policy_type='vanila', checkpoint_dir="save_model", name=None):
        """
        policy_type = {'vanila', 'goal'}
        """
        self.action_dim = action_dim
        if policy_type == 'goal':
            self.q_net = Qnet(state_dim=state_dim * 2, hidden_dim=hidden_dim, action_dim=action_dim, name="q_net", checkpoint_dir=checkpoint_dir).to(device)
            self.target_q_net = Qnet(state_dim=state_dim * 2, hidden_dim=hidden_dim, action_dim=action_dim, name="target_q_net", checkpoint_dir=checkpoint_dir).to(device)
        elif policy_type == 'vanila':
            self.q_net = Qnet(state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim, name="q_net", checkpoint_dir=checkpoint_dir).to(device)
            self.target_q_net = Qnet(state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim, name="target_q_net", checkpoint_dir=checkpoint_dir).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.device = device
        self.length = length
        self.cnt = 0
        self.type = policy_type


    def take_action(self, state, goal=None):
        if self.type == 'goal':
            assert goal is not None
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.length)
        else:
            if self.type == 'goal':
                state = np.concatenate([state, goal])
            state = torch.tensor([state], dtype=torch.float32).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def learn(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float32).to(self.device)
        # ipdb.set_trace()
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float32).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float32).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float32).view(-1, 1).to(self.device)
        goals = torch.tensor(transition_dict['goals'], dtype=torch.float32).to(self.device)

        # ipdb.set_trace()
        states = torch.cat([states, goals], dim=1)
        next_states = torch.cat([next_states, goals], dim=1)

        q_values = self.q_net(states).gather(1, actions)
        q_values = q_values.cuda()
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        # dqn_loss.backward(retain_graph=True)
        self.optimizer.step()

        if self.cnt % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.cnt += 1

    def save_model(self):
        self.q_net.save_checkpoint()
        self.target_q_net.save_checkpoint()

    def load_model(self):
        self.q_net.load_checkpoint()
        self.target_q_net.load_checkpoint()
        
