import numpy as np
from src.ReplayBuffer import ReplayBuffer
from typing import Callable
import torch
from src.stats import StatsRecorder
import os
import types

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
                 epsilon:float|Callable=0.1, 
                 alpha:int=0.1, 
                 MAX_STEPS:int=200, 
                 MAX_EPISODES:int=1, 
                 BATCH_SIZE:int=64,
                 stats:StatsRecorder=None,
                 model_folder:str="models",
                 export_frequency:int=1_000,
                 eval = False,
                 use_target_network = False,
                 target_update_freq = 10_000):
        
        self.num_actions = num_actions
        self.dim = obs_dim * 2 + 3 # state + action + next_state + reward + done 
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
        self.Q_avg = 0
        self.Q_head_right_avg = 0
        self.Q_head_stay_avg = 0
        self.Q_head_left_avg = 0
        self.right_counter = 0
        self.stay_counter = 0
        self.left_counter = 0
        self.mean_loss = 0

        network_width = 32

        # check if gpu available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the Q-network
        self.q_network = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, network_width),
            torch.nn.ReLU(),
            torch.nn.Linear(network_width, network_width),
            torch.nn.ReLU(),
            torch.nn.Linear(network_width, num_actions)
        ).to(self.device)
        self.loss = torch.nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.alpha)
        # torch.nn.init.xavier_uniform_(self.q_network[0].weight)
        self.use_target_network = use_target_network
        self.target_network_update_freq = target_update_freq
        if use_target_network:
            self.target_q_network = torch.nn.Sequential(
                torch.nn.Linear(obs_dim, network_width),
                torch.nn.ReLU(),
                torch.nn.Linear(network_width, network_width),
                torch.nn.ReLU(),
                torch.nn.Linear(network_width, num_actions)
            ).to(self.device)

            self.target_q_network.load_state_dict(self.q_network.state_dict().copy())

        # Check for epsilon function
        if eval:
            epsilon = 0
        if not isinstance(epsilon, types.FunctionType):
            self.get_epsilon_value = lambda _: epsilon
        else:
            self.get_epsilon_value = epsilon

        # clear models folder
        if not eval:
            if os.path.exists(model_folder):
                print("Clearing models folder...", end="", flush=True)
                for file in os.listdir(model_folder):
                    os.remove(os.path.join(model_folder, file))
                print("DONE")
            else:
                os.makedirs(model_folder)
        

    def observe(self, state, action, next_state, reward, done=False):
        self.replay_buffer.add(state, action, next_state, reward, done)

    @torch.no_grad()
    def select_action(self, state, iter:int=0):
        e = self.get_epsilon_value(iter)
        if self.stats is not None:
            self.stats.record_epsilon(e)
        if np.random.rand() < e:
            return np.random.randint(0, self.num_actions)
        else:
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            output = self.q_network(state)
            return torch.argmax(output).item()

    @torch.enable_grad()
    def update(self) -> dict[str, tuple[int, float]]:
        if self.replay_buffer.total_size < self.BATCH_SIZE:
            return
        # sample a batch from the replay buffer
        batch = self.replay_buffer.sample(self.BATCH_SIZE)
        batch = torch.tensor(batch, dtype=torch.float32, device=self.device)
        states, actions, next_states, rewards, dones = torch.split(batch, split_size_or_sections=[self.obs_dim, 1, self.obs_dim, 1, 1], dim=1)
        actions = actions.squeeze(1)
        rewards = rewards.squeeze(1)
        done_mask = 1 - dones
        with torch.no_grad():
            if self.use_target_network:
                next_Q_values = self.target_q_network(next_states) * done_mask
            else:
                next_Q_values = self.q_network(next_states) * done_mask
            target_Q_values = next_Q_values.max(dim=1).values * self.discount + rewards
        self.optimizer.zero_grad() # zero the gradients before the forward pass
        #Forward pass
        Q_values = self.q_network(states)[torch.arange(self.BATCH_SIZE), actions.type(torch.int64)]
        loss = self.loss(Q_values, target_Q_values)
        loss.backward()
        self.optimizer.step()

        # Log the training data
        if self.stats is not None:
            Q_head_right = Q_values[actions == 2]
            Q_head_stay = Q_values[actions == 1]
            Q_head_left = Q_values[actions == 0]
            Q_avg = Q_values.mean().item()
            Q_head_right_avg = Q_head_right.mean()
            Q_head_stay_avg = Q_head_stay.mean()
            Q_head_left_avg = Q_head_left.mean()
            self.Q_avg = (self.Q_avg*(self.update_counter) + Q_avg) / (self.update_counter + 1)
            if not torch.isnan(Q_head_right_avg):
                self.Q_head_right_avg = (self.Q_head_right_avg*(self.right_counter) + Q_head_right_avg.item()) / (self.right_counter + 1)
                self.right_counter += 1
            if not torch.isnan(Q_head_stay_avg):
                self.Q_head_stay_avg = (self.Q_head_stay_avg*(self.stay_counter) + Q_head_stay_avg.item()) / (self.stay_counter + 1)
                self.stay_counter += 1
            if not torch.isnan(Q_head_left_avg):
                self.Q_head_left_avg = (self.Q_head_left_avg*(self.left_counter) + Q_head_left_avg.item()) / (self.left_counter + 1)
                self.left_counter += 1
            
            self.mean_loss = (self.mean_loss*(self.update_counter) + float(loss.item())) / (self.update_counter + 1)
            self.update_counter += 1
            self.stats.log(**{
                "training/loss": (self.update_counter, float(loss.item())),
                "training/loss_mean": (self.update_counter, self.mean_loss),
                "training/mean_Q": (self.update_counter, float(self.Q_avg)),
                "training/Q_head_right": (self.right_counter, float(self.Q_head_right_avg)),
                "training/Q_head_stay": (self.stay_counter, float(self.Q_head_stay_avg)),
                "training/Q_head_left": (self.left_counter, float(self.Q_head_left_avg)),
        })

            
        if self.update_counter % self.export_frequency == 0:
            self.save()
        
        if self.use_target_network and self.update_counter % self.target_network_update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict().copy())
            
    def save(self):
        name = f"dqn_model_{self.update_counter}.pth"
        path = self.model_folder + "/" + name
        torch.save(self.q_network.state_dict(), path)

    def load(self, path):
        self.q_network.load_state_dict(torch.load(path, map_location=self.device))
        self.q_network.eval()
    
    @torch.no_grad()
    def get_q_values(self, states):
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        return self.q_network(states).to("cpu").numpy()


class DynaAgent():
    def __init__(self):
        pass

    def observe(self, state, action, next_state, reward):
        pass

    def select_action(self, state):
        pass

    def update(self):
        pass