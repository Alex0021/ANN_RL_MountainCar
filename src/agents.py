import numpy as np
from src.ReplayBuffer import ReplayBuffer

class RandomAgent():
    def __init__(self, num_actions:int, obs_dim:int, MAX_STEPS:int=200):
        self.num_actions = num_actions
        self.obs_dim = obs_dim * 2 + 2
        self.replay_buffer = ReplayBuffer(self.obs_dim, MAX_STEPS)

    def observe(self, state, action, next_state, reward, done=False):
        self.replay_buffer.add(state, action, next_state, reward, done)

    def select_action(self, state):
        return np.random.randint(0, self.num_actions)

    def update(self):
        pass


class DqnAgent():
    def __init__(self):
        pass

    def observe(self, state, action, next_state, reward):
        pass

    def select_action(self, state):
        pass

    def update(self):
        pass


class DynaAgent():
    def __init__(self):
        pass

    def observe(self, state, action, next_state, reward):
        pass

    def select_action(self, state):
        pass

    def update(self):
        pass