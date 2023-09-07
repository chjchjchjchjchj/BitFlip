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
import rl_utils

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transition = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transition)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # ipdb.set_trace()
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device, length):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim).to(device)

        self.target_q_net = Qnet(state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim).to(device)

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.device = device
        self.length = length
        self.cnt = 0

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.length)
        else:
            state = torch.tensor([state], dtype=torch.float32).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float32).to(self.device)
        # ipdb.set_trace()
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float32).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float32).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float32).view(-1, 1).to(self.device)
        # ipdb.set_trace()
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


def set_seed(seed, env):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)

def main():
    length = 4
    lr = 1e-3
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.99
    epsilon = 0.01
    target_update = 10
    buffer_size = 10000
    batch_size = 64
    step_is_fast = True
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    env_name = "BitFlip"
    env = BitFlip(length=length, step_is_fast=step_is_fast)

    # env_name = 'CartPole-v0'
    # env = gym.make(env_name)

    set_seed(10, env=env)
    minimal_size = 500
    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQN(state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim, learning_rate=lr, gamma=gamma, epsilon=epsilon, target_update=target_update, device=device, length=length)

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i}') as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state, _ = env.reset()
                # state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    # next_state, reward, done = env.step(action)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward

                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix(
                        {
                            'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                            'return': '%.3f' % np.mean(return_list[-10:])
                        }
                    )
                    pbar.update(1)


        episodes_list = list(range(len(return_list)))
        plt.plot(episodes_list, return_list)
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title('DQN on {}'.format(env_name))
        plt.savefig('DQN on {}(bf).pdf'.format(env_name))
        plt.show()

        mv_return = rl_utils.moving_average(return_list, 9)
        plt.plot(episodes_list, mv_return)
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title('DQN on {}'.format(env_name))
        plt.savefig('DQN on {}(m).pdf'.format(env_name))
        plt.show()

if __name__ == "__main__":
    main()   