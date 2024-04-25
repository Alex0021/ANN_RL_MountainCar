import numpy as np
from src.ReplayBuffer import ReplayBuffer
from typing import Callable
import torch
from src.stats import StatsRecorder
import os

class RandomAgent():
    def __init__(self, num_actions:int, obs_dim:int, MAX_STEPS:int=200, MAX_EPISODES:int=1):
        self.num_actions = num_actions
        self.dim = obs_dim * 2 + 2
        self.replay_buffer = ReplayBuffer(self.dim, MAX_STEPS, MAX_EPISODES)

    def observe(self, state, action, next_state, reward, done=False):
        self.replay_buffer.add(state, action, next_state, reward, done)

    def select_action(self, state):
        return np.random.randint(0, self.num_actions)

    def update(self):
        pass


class DqnAgent():
    def __init__(self, num_actions:int, 
                 obs_dim:int, 
                 discount:float=0.99, 
                 epsilon:float|Callable=0.99, 
                 alpha:int=0.1, 
                 MAX_STEPS:int=200, 
                 MAX_EPISODES:int=1, 
                 BATCH_SIZE:int=64,
                 stats:StatsRecorder=None,
                 model_folder:str="models",
                 export_frequency:int=1000,
                 eval = False):
        
        self.num_actions = num_actions
        self.dim = obs_dim * 2 + 2
        self.obs_dim = obs_dim
        self.discount = discount
        self.epsilon = epsilon
        self.alpha = alpha
        self.replay_buffer = ReplayBuffer(self.dim, MAX_STEPS, MAX_EPISODES)
        self.BATCH_SIZE = BATCH_SIZE
        self.update_counter = 0
        self.stats = stats
        self.model_folder = model_folder
        self.export_frequency = export_frequency

        # Initialize the Q-network
        self.q_network = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 64),
            torch.nn.Linear(64, 64),
            torch.nn.Linear(64, num_actions)
        )
        self.loss = torch.nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.alpha)
        # torch.nn.init.xavier_uniform_(self.q_network[0].weight)

        # clear models folder
        if not eval:
            if os.path.exists(model_folder):
                for file in os.listdir(model_folder):
                    os.remove(os.path.join(model_folder, file))
            else:
                os.makedirs(model_folder)
        



    def observe(self, state, action, next_state, reward, done=False):
        self.replay_buffer.add(state, action, next_state, reward, done)

    @torch.no_grad()
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.num_actions)
        else:
            state = torch.tensor(state, dtype=torch.float32)
            return torch.argmax(self.q_network(state))

    @torch.enable_grad()
    def update(self) -> dict[str, tuple[int, float]]:
        if self.replay_buffer.total_size < self.BATCH_SIZE:
            return
        # sample a batch from the replay buffer
        batch = self.replay_buffer.sample(self.BATCH_SIZE)
        batch = torch.tensor(batch, dtype=torch.float32)
        self.optimizer.zero_grad()
        #Forward pass
        states, actions, next_states, rewards = torch.split(batch, split_size_or_sections=[self.obs_dim, 1, self.obs_dim, 1], dim=1)
        Q_values = self.q_network(states)[:, actions.type(torch.int64)]
        target_Q_values = self.q_network(next_states).max(dim=1).values * self.discount + rewards
        loss = self.loss(Q_values, target_Q_values)
        loss.backward()
        self.optimizer.step()

        # Log the training data
        if self.stats is not None:
            self.update_counter += 1
            self.stats.log(**{
                "training/loss": (self.update_counter, float(loss.item())),
                "training/mean_Q": (self.update_counter, float(Q_values.mean().item()))
        })
            
        if self.update_counter % self.export_frequency == 0:
            self.save()
            
    def save(self):
        name = f"dqn_model_{self.update_counter}.pth"
        path = self.model_folder + "/" + name
        torch.save(self.q_network.state_dict(), path)

    def load(self, path):
        self.q_network.load_state_dict(torch.load(path))


class DynaAgent():
    def __init__(self):
        pass

    def observe(self, state, action, next_state, reward):
        pass

    def select_action(self, state):
        pass

    def update(self):
        pass