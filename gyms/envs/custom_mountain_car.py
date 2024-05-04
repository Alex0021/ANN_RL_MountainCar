from gymnasium.envs.classic_control.mountain_car import MountainCarEnv
import numpy as np

class CustomMountainCar(MountainCarEnv):

    def __init__(self, **kwargs):
        super(CustomMountainCar, self).__init__(**kwargs)

    def step(self, action):
        state, reward, terminated, truncated, _ = super(CustomMountainCar, self).step(action)
        height = np.sin(3 * (state[0] + (np.pi / 3 - 0.5)) - np.pi/2) 
        reward += height 
        reward += 5*np.abs(state[1])
        return state, reward, terminated, truncated, _