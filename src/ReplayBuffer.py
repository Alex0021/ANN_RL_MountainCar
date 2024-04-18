import numpy as np

class ReplayBuffer():
    
    def __init__(self, obs_dim:int, MAX_STEPS:int, MAX_EPISODES:int=1):
        self.MAX_EPISODES = MAX_EPISODES
        self.MAX_STEPS = MAX_STEPS
        self.obs_dim = obs_dim # state + action + next_state + reward (so state_dim * 2 + 2)
        self.buffer = np.zeros((MAX_EPISODES, MAX_STEPS, obs_dim))
        # self.episode_info = np.zeros((MAX_EPISODES, 1)) # (step_ptr) 
        self.episode_ptr = 0
        self.step_ptr = 0
        self.size = 0


    def add(self, state:np.ndarray, action:int, next_state:np.ndarray, reward:float, done:bool):
        if self.step_ptr >= self.MAX_STEPS:
            raise ValueError("maximum episode length exceeded.")
        
        # if the current episode buffer is not empty, zero it out
        if self.step_ptr == 0 and not np.all(self.buffer[self.episode_ptr] == 0):
            print(f"BUFFER IS FULL: Zeroing out episode buffer at index {self.episode_ptr}")

            # subtract the number of zeroed out steps from the size
            nonzero = np.nonzero(self.buffer[self.episode_ptr])
            last_step = nonzero[0][-1]
            self.size -= last_step + 1

            self.buffer[self.episode_ptr] = np.zeros((self.MAX_STEPS, self.obs_dim))
        
        self.buffer[self.episode_ptr][self.step_ptr] = np.concatenate([state, np.array([action]), next_state, np.array([reward])])
        
        if done:
            self.episode_ptr = (self.episode_ptr + 1) % self.MAX_EPISODES
            self.step_ptr = 0
        else:
            self.step_ptr += 1
            
        self.size += 1

    def sample(self, batch_size:int):
        # don't forget to mask the zero padding
        pass

    def __len__(self):
        return self.size

    




