import gym
import numpy as np
from gym import spaces
import ipdb

class BitFlip(gym.Env):
    def __init__(self, render_mode=None, length=5, reward_type="default"):
        super(BitFlip, self).__init__()
        """
        reward_type = ["default", "euclidean"]
        """
        self.length = length
        self.action_space = spaces.Discrete(self.length)
        self.observation_space = spaces.Box(0, 1, [self.length], dtype=np.float32)
        self.goal = np.random.randint(2, size=(self.length))
        self.state = np.random.randint(2, size=(self.length))
        self.reward_type=reward_type

    def reset(self):
        self.goal = np.random.randint(2, size=(self.length))
        self.state = np.random.randint(2, size=(self.length))

        return np.copy(self.state), np.copy(self.goal)

    def step(self, action):
        # ipdb.set_trace()
        # print(f"action={action}")
        self.state[action] = 1 - self.state[action]
        done = np.array_equal(self.state, self.goal)
        if self.reward_type == "default":
            reward = 1.0 if done else 0
        elif self.reward_type == "euclidean":
            reward = -np.linalg.norm(self.state - self.goal)

        return np.copy(self.state), reward, done, None
    
    def render(self):
        # print(self.state)
        print(f"Current State: {self.state}")
    