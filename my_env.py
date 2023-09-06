import gym
import numpy as np
from gym import spaces
import ipdb

class BitFlip(gym.Env):
    def __init__(self, render_mode=None, length=5, reward_type="default", reward_success=1, reward_fail=0, step_is_fast=False):
        super(BitFlip, self).__init__()
        self.length = length
        if step_is_fast:
            self.action_space = spaces.Box(0, 1, [self.length], dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(self.length)
        self.observation_space = spaces.Box(0, 1, [self.length], dtype=np.float32)
        self.goal = np.random.randint(2, size=(self.length))
        self.state = np.random.randint(2, size=(self.length))
        self.reward_type = reward_type
        self.reward_success = reward_success
        self.reward_fail = reward_fail
        self.step_is_fast = step_is_fast

    def reset(self):
        self.goal = np.random.randint(2, size=(self.length))
        self.state = np.random.randint(2, size=(self.length))

        return np.copy(self.state), np.copy(self.goal)

    def step(self, action):
        # ipdb.set_trace()
        # print(f"action={action}")
        if self.step_is_fast:
            self.state[action == 1] = 1 - self.state[action == 1]
        else:
            self.state[action] = 1 - self.state[action]
        done = np.array_equal(self.state, self.goal)
        if self.reward_type == "default":
            reward = self.reward_success if done else self.reward_fail
        elif self.reward_type == "euclidean":
            reward = -np.linalg.norm(self.state - self.goal)

        return np.copy(self.state), reward, done, None  
    
    def render(self):
        # print(self.state)
        print(f"Current State: {self.state}")
