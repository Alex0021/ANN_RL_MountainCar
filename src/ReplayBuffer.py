import numpy as np
from collections import defaultdict

class ReplayBuffer():
    """
    dim: state + action + next_state + reward (so state_dim * 2 + 2)
    """
    
    def __init__(self, dim:int, MAX_STEPS:int, MAX_EPISODES:int=1):
        self.MAX_EPISODES = MAX_EPISODES
        self.MAX_STEPS = MAX_STEPS
        self.dim = dim # state + action + next_state + reward + done (so state_dim * 2 + 3)
        self.buffer = np.zeros((MAX_EPISODES*MAX_STEPS, dim))
        self.mapping = defaultdict() # (episode_ptr, step_ptr)
        # self.episode_info = np.zeros((MAX_EPISODES, 1)) # (step_ptr) 
        self.episode_ptr = 0
        self.step_ptr = 0
        self.index = 0
        self.total_size = 0


    def add(self, state:np.ndarray, action:int, next_state:np.ndarray, reward:float, done:bool):
        if self.step_ptr >= self.MAX_STEPS:
            raise ValueError("maximum episode length exceeded.")
        
        self.mapping[(self.episode_ptr, self.step_ptr)] = self.index
        self.buffer[self.index] = np.concatenate([state, np.array([action]), next_state, np.array([reward]), np.array([done])])
        self.index = (self.index+1) % (self.MAX_EPISODES * self.MAX_STEPS)
        
        if done:
            self.episode_ptr = (self.episode_ptr + 1) % self.MAX_EPISODES
            self.step_ptr = 0
        else:
            self.step_ptr = self.step_ptr + 1

        self.total_size = min(self.total_size + 1, self.MAX_EPISODES * self.MAX_STEPS)
            

    def sample(self, batch_size:int):
        if self.total_size < batch_size:
            raise ValueError("Not enough samples to sample from.")
        indices = np.random.randint(0, self.total_size, batch_size)
        samples = self.buffer[indices]
        return samples

    def __len__(self):
        return self.size

    




