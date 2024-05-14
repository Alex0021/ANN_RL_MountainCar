from gymnasium.envs.classic_control.mountain_car import MountainCarEnv
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib import pyplot as plt

class CustomMountainCar(MountainCarEnv):

    def __init__(self, **kwargs):
        super(CustomMountainCar, self).__init__(**kwargs)
        self.max_r = -np.inf
        self.max_l = -np.inf
        self.middle_x = -np.pi/6
        self.fig = plt.figure(num=1)

    def step(self, action):
        state = self.state
        next_state, reward, terminated, truncated, _ = super(CustomMountainCar, self).step(action)
        # reward = 1 if reward 
        # n_height = self.normed_height(state[0])
        # pos_sign = 1 if state[0] > self.middle_x else -1
        # action_sign = action - 1
        # reward += n_height * -1 * pos_sign * action_sign

        # reward += self.normed_height(state[0]) 
        reward += 1 if terminated else 0
        speed  = abs(state[1])
        normed_speed = speed / 0.07
        aux_reward = normed_speed
        # reward += self.max_height_reward(state)
        aux_reward += self.normed_height(state[0])**2
        return next_state, reward+aux_reward, terminated, truncated, {"aux_reward": aux_reward, "env_reward": reward}
    
    def max_height_reward(self, state):
        n_height = self.normed_height(state[0])
        if state[0] > self.middle_x and n_height > self.max_r:
            self.max_r = n_height
            return 0.5
        if state[0] < self.middle_x and n_height > self.max_l:
            self.max_l = n_height
            return 0.5
        return 0


    def normed_height(self, x):
        height = self.curve(x) + 1 
        normed_height = height / 2
        return normed_height
    
    def curve(self, x):
        return np.sin(3 * x)
    
    def d_curve(self, x):
        return 3 * np.cos(3 * x)

    def plot_state(self, state, reward, agent=None):
        plt.ion()
        x = np.linspace(self.min_position, self.max_position, 100)
        # curve = lambda x: np.cos(3 * (x + (np.pi / 3 - 0.5)))
        y = self.curve(x)
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        if agent:
            q_values = agent.get_q_values([[x, state[1]] for x in x])
            v_values = np.mean(q_values, axis=1)
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(v_values.min(), v_values.max())
            lc = LineCollection(segments, cmap='jet', norm=norm)
            # Set the values used for colormapping
            lc.set_array(v_values)
            lc.set_linewidth(2)
            line = ax.add_collection(lc)
            cb = self.fig.colorbar(line, ax=ax)
            cb.set_label('V-value')
        else:
            ax.plot(x, y)
        ax.scatter(state[0], self.curve(state[0]), s=100, c='k', marker='o')
        text = f"Reward: {reward:.4f}"
        self.fig.suptitle(text, fontsize=12)
        # v line at middle
        ax.axvline(x=self.middle_x, color='r')
        plt.draw()
        plt.pause(0.001)

