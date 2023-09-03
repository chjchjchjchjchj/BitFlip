import gym
import numpy as np
from gym import spaces
import ipdb

class BitFlip(gym.Env):
    def __init__(self, render_mode=None, length=5):
        super(BitFlip, self).__init__()
        self.length = length
        self.action_space = spaces.Discrete(self.length)
        # self.action_space = spaces.Box(low=0, high=self.length)
        self.observation_space = spaces.Box(0, 1, [self.length], dtype=np.float32)
        # self.observation_space = spaces.Discrete()

    def reset(self):
        self.goal = np.random.randint(2, size=(self.length))
        self.state = np.random.randint(2, size=(self.length))

        return self.state, self.goal

    def step(self, action):
        # ipdb.set_trace()
        # print(f"action={action}")
        self.state[action] = 1 - self.state[action]
        done = np.array_equal(self.state, self.goal)
        
        reward = 1.0 if done else 0

        return self.state, reward, done, None
    
    def render(self):
        print(self.state)
    
