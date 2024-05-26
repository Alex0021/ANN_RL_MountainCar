import numpy as np
from src.ReplayBuffer import ReplayBuffer
from typing import Callable
import torch
from src.stats import StatsRecorder
import os
import types

class Agent():
    def __init__(self):
        pass 
    
    def observe(self):
        pass

    def select_action(self):
        pass

    def update(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def clear_models(self):
        pass

    def export_model(self):
        pass

class RandomAgent(Agent):
    def __init__(self, num_actions:int, obs_dim:int, MAX_STEPS:int=200, MAX_EPISODES:int=1, stats:StatsRecorder=None):
        self.num_actions = num_actions
        self.dim = obs_dim * 2 + 4
        self.replay_buffer = ReplayBuffer(self.dim, MAX_STEPS, MAX_EPISODES)
        self.stats = stats

    def observe(self, state, action, next_state, reward, aux_reward, done=False):
        # Save reward and action
        self.stats.record_action(action)
        self.stats.record_reward(reward, aux_reward)
        obs = np.concatenate([state, np.array([action]), next_state, [reward, aux_reward], np.array([done])])
        self.replay_buffer.add(obs, done)

    def select_action(self, state, iter:int=0):
        return np.random.randint(0, self.num_actions)

class DqnAgent(Agent):

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
                 use_target_network = True,
                 target_update_freq = 10_000,
                 STEPS_DELAY:int=1_000):
        
        self.num_actions = num_actions
        self.dim = obs_dim * 2 + 4 # state + action + next_state + reward + aux_reward + done 
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
        self.STEPS_DELAY = STEPS_DELAY

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
                print("Clearing models folder...")
                self.clear_models(model_name=self.__class__.__name__)
            else:
                os.makedirs(model_folder)

    def observe(self, state, action, next_state, reward, aux_reward, done=False):
        # Save reward and action
        self.stats.record_action(action)
        self.stats.record_reward(reward, aux_reward)
        obs = np.concatenate([state, np.array([action]), next_state, [reward, aux_reward], np.array([done])])
        # Store to buffer
        self.replay_buffer.add(obs, done)

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
        if self.replay_buffer.total_size < self.BATCH_SIZE or self.stats.total_steps < self.STEPS_DELAY:
            return
        # sample a batch from the replay buffer
        batch = self.replay_buffer.sample(self.BATCH_SIZE)
        batch = torch.tensor(batch, dtype=torch.float32, device=self.device)
        states, actions, next_states, rewards, dones = torch.split(batch, split_size_or_sections=[self.obs_dim, 1, self.obs_dim, 2, 1], dim=1)
        actions = actions.squeeze(1)
        rewards = rewards.sum(dim=1)
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
        self._save_update_stats(Q_values, actions, loss)
            
        if self.update_counter % self.export_frequency == 0:
            self.save(model_name=self.__class__.__name__)
        
        if self.use_target_network and self.update_counter % self.target_network_update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict().copy())

    def _save_update_stats(self, Q_values, actions, loss):
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
            
    def save(self, model_name:str='agentXXX'):
        name = f"{model_name}_model_{self.update_counter}.pth"
        path = self.model_folder + "/" + name
        torch.save(self.q_network.state_dict(), path)

    def load(self, path):
        self.q_network.load_state_dict(torch.load(path, map_location=self.device))
        self.q_network.eval()
    
    def clear_models(self, model_name:str='agentXXX'):
        for file in os.listdir(self.model_folder):
            if file.startswith(model_name):
                os.remove(os.path.join(self.model_folder, file))
    
    @torch.no_grad()
    def get_q_values(self, states):
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        return self.q_network(states).to("cpu").numpy()

    def export_model(self, folder_path):
        name = "model.pth"
        path = folder_path + "/" + name
        torch.save(self.q_network.state_dict(), path)
    
class DqnAgentRND(DqnAgent):
    def __init__(self, num_actions:int, 
                 obs_dim:int, 
                 discount:float=0.99, 
                 epsilon:float|Callable=0.1, 
                 alpha:int=0.1, 
                 MAX_STEPS:int=200, 
                 MAX_EPISODES:int=1, 
                 BATCH_SIZE:int=64,
                 reward_factor:float|Callable=2.0,
                 RND_NORMALIZE_DELAY:int=5,
                 rnd_alpha:float=0.1,
                 stats:StatsRecorder=None,
                 model_folder:str="models",
                 export_frequency:int=1_000,
                 eval:bool=True,
                 use_target_network:bool=False,
                 target_update_freq:int=10_000):
        
        super().__init__(num_actions, obs_dim, discount, epsilon, alpha, 
                         MAX_STEPS, MAX_EPISODES, BATCH_SIZE, stats, model_folder, 
                         export_frequency, eval, use_target_network, target_update_freq)
        
        self.RND_NORMALIZE_DELAY = RND_NORMALIZE_DELAY*MAX_STEPS

        self.rnd_reward = RNDreward(obs_dim, rnd_alpha, stats)

        if not isinstance(reward_factor, types.FunctionType):
            rw_f = lambda _: reward_factor
        else:
            rw_f = reward_factor
            
        self.reward_factor = rw_f


    @torch.enable_grad()
    def update(self) -> dict[str, tuple[int, float]]:
        if self.replay_buffer.total_size < self.BATCH_SIZE or self.stats.total_steps < self.STEPS_DELAY:
            return
        # Update Q-network
        # sample a batch from the replay buffer
        batch = self.replay_buffer.sample(self.BATCH_SIZE)
        batch = torch.tensor(batch, dtype=torch.float32, device=self.device)
        states, actions, next_states, rewards, dones = torch.split(batch, split_size_or_sections=[self.obs_dim, 1, self.obs_dim, 2, 1], dim=1)
        actions = actions.squeeze(1)
        # Normalize aux reward
        aux_rewards = self.rnd_reward.normalize_and_clamp_reward(rewards[:,1])
        # Find total rewards
        rewards = rewards[:,0] + aux_rewards * self.reward_factor(self.update_counter)
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

        # Update RND network
        if self.stats.total_steps > self.RND_NORMALIZE_DELAY:
            self.rnd_reward.update(next_states)
            
        # Log the training data
        self._save_update_stats(Q_values, actions, loss)
        
        if self.update_counter % self.export_frequency == 0:
            self.save(model_name=self.__class__.__name__)
        
        if self.use_target_network and self.update_counter % self.target_network_update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict().copy())

        if self.stats is not None:
            self.stats.log(**{
                "training/reward_factor": (self.update_counter, self.reward_factor(self.update_counter)),
            })

    @torch.no_grad()
    def observe(self, state, action, next_state, reward, aux_reward, done=False):

        # Overwriting aux reward
        aux_reward = self.rnd_reward.observe(next_state)

        self.stats.record_action(action)

        obs = np.concatenate([state, np.array([action]), next_state, [reward, aux_reward], np.array([done])])
        
        self.replay_buffer.add(obs, done)
        
        # Save aux_reward to tensorboard
        # Use estimate of current normalization factors
        aux_reward_norm = self.rnd_reward.normalize_and_clamp_reward(aux_reward) * self.reward_factor(self.update_counter) if self.stats.total_steps > 1 else aux_reward
        self.stats.record_reward(reward, aux_reward_norm)
        
class RNDreward():
    def __init__(self, obs_dim:int, rnd_alpha:float=0.1, stats:StatsRecorder=None):
        self.obs_dim = obs_dim
        self.state_mean = np.zeros(obs_dim)
        self.state_var = np.zeros(obs_dim)
        self.reward_mean = 0
        self.reward_var = 0
        self.update_state_counter = 0
        self.update_reward_counter = 0
        self.update_counter = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.stats = stats

        # Initialize RND networks
        LAYER_SIZE = 32
        self.target = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, LAYER_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(LAYER_SIZE, LAYER_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(LAYER_SIZE, 1)
        ).to(self.device)

        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, LAYER_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(LAYER_SIZE, LAYER_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(LAYER_SIZE, 1)
        ).to(self.device)

        self.predictor_optimizer = torch.optim.Adam(self.predictor.parameters(), lr=rnd_alpha)
        self.predictor_loss = torch.nn.MSELoss(reduction='mean')

    @torch.no_grad()
    def observe(self, state):

        # Update state statistics
        old_state_mean = self.state_mean.copy()
        self.state_mean = (self.state_mean*self.update_state_counter + state) / (self.update_state_counter + 1)
        self.state_var = (self.state_var*self.update_state_counter + (state - self.state_mean)*(state - old_state_mean)) / (self.update_state_counter + 1)
        self.update_state_counter += 1

        # compute reward
        norm_state = self.normalize_state(state) if self.update_state_counter > 1 else state
        input = torch.tensor(norm_state, device=self.device, dtype=torch.float32)
        pred = self.predictor(input)
        target = self.target(input)
        reward = self.predictor_loss(pred, target).item()
        
        # update reward statistics
        old_reward_mean = self.reward_mean
        self.reward_mean = (self.reward_mean*self.update_reward_counter + reward) / (self.update_reward_counter + 1)
        self.reward_var = (self.reward_var*self.update_reward_counter + (reward - self.reward_mean)*(reward - old_reward_mean)) / (self.update_reward_counter + 1)
        self.update_reward_counter += 1

        # log to tensorboard
        if self.update_reward_counter > 1:
            if self.stats is not None:
                self.stats.log(**{
                    "RND/reward": (self.update_reward_counter, reward),
                    "RND/reward_mean": (self.update_reward_counter, self.reward_mean),
                    "RND/reward_var": (self.update_reward_counter, self.reward_var),
                    "RND/reward_norm": (self.update_reward_counter, self.normalize_and_clamp_reward(reward) if self.update_reward_counter > 1 else reward),
                    "RND/state_mean_0": (self.update_state_counter, self.state_mean[0]),
                    "RND/state_var_0": (self.update_state_counter, self.state_var[0]),
                    "RND/state_mean_1": (self.update_state_counter, self.state_mean[1]),
                    "RND/state_var_1": (self.update_state_counter, self.state_var[1])
                })

        return reward


    @torch.enable_grad()
    def update(self, next_states:torch.Tensor):
        # Update RND network
        with torch.no_grad():
            norm_states = self.normalize_state(next_states)
            norm_states = torch.tensor(norm_states, device=self.device, dtype=torch.float32)
            target = self.target(norm_states)
        self.predictor_optimizer.zero_grad()
        pred = self.predictor(norm_states)
        loss = self.predictor_loss(pred, target)
        loss.backward()
        self.predictor_optimizer.step()

        self.update_counter += 1

        if self.stats is not None:
            self.stats.log(**{
                "RND/loss": (self.update_counter, loss.item()),
            })

    def normalize_reward(self, rewards):
        return (rewards - self.reward_mean) / np.sqrt(self.reward_var)
    
    def normalize_state(self, state):
        return (state - self.state_mean) / np.sqrt(self.state_var)

    def normalize_and_clamp_reward(self, rewards):
        rw = (rewards - self.reward_mean) / np.sqrt(self.reward_var)
        return np.clip(rw, -5, 5)

class DynaAgent():
    def __init__(self,
                 num_actions:int, 
                 obs_dim:int, 
                 discount:float=0.99, 
                 epsilon:float|Callable=0.1,
                 lr:float=0.1,
                 MAX_STEPS:int=200,
                 MAX_EPISODES:int=1,
                 eval:bool=True,
                 export_frequency:int=1_000,
                 k:int=10,
                 model_folder:str="models",
                 stats:StatsRecorder=None,
                 n_bins:int=(100, 100),
                 action_ranges:list[list]=[[-1.2, 0.6], [-0.07, 0.07]]):
        
        self.num_actions = num_actions
        self.dim = obs_dim + 2 # state + action
        self.obs_dim = obs_dim
        self.discount = discount
        self.epsilon = epsilon
        self.lr = lr
        self.replay_buffer = ReplayBuffer(self.dim, MAX_STEPS, MAX_EPISODES)
        self.k = k
        self.stats = stats
        self.model_folder = model_folder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_ranges = np.diff(np.array(action_ranges), axis=1).flatten()
        self.actions_min = np.array(action_ranges)[:,0]
        self.n_bins = np.array(n_bins)
        self.bin_sizes = np.divide(self.action_ranges, self.n_bins+1)
        n_states = np.prod(self.n_bins+1)
        self.update_counter = 0
        self.export_frequency = export_frequency
        self.eval = eval
        
        self.Q_table = np.zeros((n_states, num_actions))
        self.R_table = np.zeros((n_states, num_actions))
        self.P_table = np.zeros((n_states, num_actions, n_states))
        self.bin_count_table = np.zeros((n_states, num_actions))

        # Stats
        self.Q_avg = 0
        self.Q_head_right_avg = 0
        self.Q_head_stay_avg = 0
        self.Q_head_left_avg = 0

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
                print("Clearing models folder...")
                self.clear_models(model_name=self.__class__.__name__)
            else:
                os.makedirs(model_folder)

    def state_to_idx_map(self, state):
        if state.ndim == 1:
            state = state.reshape(1, -1)
        idx = np.floor(np.divide(state-self.actions_min, self.bin_sizes)) 
        idx = np.clip(idx, 0, self.n_bins)
        idx = idx[:,0]*self.n_bins[1] + idx[:,1]
        return idx.astype(int)
    
    def Q(self, state, action):
        idx = self.state_to_idx_map(state)
        return self.Q_table[idx, int(action)]
    
    def R(self, state, action):
        idx = self.state_to_idx_map(state)
        return self.R_table[idx, int(action)]
    
    def P(self, state, action, next_state):
        idx0 = self.state_to_idx_map(state)
        idx1 = self.state_to_idx_map(next_state)
        return self.P_table[idx0, int(action), idx1]
    
    def observe(self, state, action, next_state, reward, aux_reward, done=False):
        # Save reward and action
        self.stats.record_action(action)
        self.stats.record_state(state)
        self.stats.record_reward(reward, 0)
        if self.eval: return

        self.replay_buffer.add(np.concatenate([state, [action, done]]), done)
        state_idx = self.state_to_idx_map(state)
        next_state_idx = self.state_to_idx_map(next_state)
        self.P_table[state_idx, action, next_state_idx] += 1

        # Running mean of rewards
        self.R_table[state_idx, action] *= self.bin_count_table[state_idx, action]
        self.R_table[state_idx, action] += reward
        self.bin_count_table[state_idx, action] += 1
        self.R_table[state_idx, action] /= self.bin_count_table[state_idx, action]

        dq = self.update_q_values(state, np.array([action]), np.array([done]))
        if self.stats is not None:
            self.stats.record_dq(dq)

    def update_q_values(self, state, action, done):
        idx = self.state_to_idx_map(state).flatten()
        idx_a = action.flatten().astype(int)
        done_mask = 1 - done
        target = self.discount * np.sum(self.P_table[idx, idx_a, :]/self.bin_count_table[idx, idx_a, np.newaxis] * np.max(self.Q_table, axis=1), axis=1) * done_mask.flatten()
        delta_q = self.lr * (self.R_table[idx, idx_a] + target - self.Q_table[idx, idx_a])
        self.Q_table[idx, idx_a] += delta_q
        return delta_q

    def select_action(self, state, iter:int=0):
        e = self.get_epsilon_value(iter)
        if self.stats is not None:
            self.stats.record_epsilon(e)
        if np.random.rand() < e:
            return np.random.randint(0, self.num_actions)
        else:
            idx = self.state_to_idx_map(state)
            return np.argmax(self.Q_table[idx])

    def update(self):
        if self.replay_buffer.total_size < self.k:
            return
        # random k samples
        k_samples = self.replay_buffer.sample(self.k)
        states, actions, dones,_ = np.hsplit(k_samples, [self.obs_dim, self.obs_dim+1, self.obs_dim+2])
        #actions = self.replay_buffer.sample_past_actions(self.k)
        self.update_q_values(states, actions, dones)

        self.update_counter += 1
        # Update stats
        self.Q_avg = np.mean(self.Q_table)
        self.Q_head_right_avg = np.mean(self.Q_table[:,2])
        self.Q_head_stay_avg = np.mean(self.Q_table[:,1])
        self.Q_head_left_avg = np.mean(self.Q_table[:,0])
        if self.stats is not None:
            self.stats.log(**{
                "training/mean_Q": (self.update_counter, self.Q_avg),
                "training/Q_head_right": (self.update_counter, self.Q_head_right_avg),
                "training/Q_head_stay": (self.update_counter, self.Q_head_stay_avg),
                "training/Q_head_left": (self.update_counter, self.Q_head_left_avg),
            })

        if self.update_counter % self.export_frequency == 0:
            self.save(model_name=self.__class__.__name__)


    def save(self, model_name:str='agentXXX'):
        name = f"{model_name}_model_{self.update_counter}.npy"
        path = self.model_folder + "/" + name
        np.save(path, self.Q_table)

    def load(self, path):
        self.Q_table = np.load(path)

    def get_q_values(self, states):
        idx = self.state_to_idx_map(np.array(states)).flatten()
        return self.Q_table[idx,:]

    def clear_models(self, model_name:str='agentXXX'):
        for file in os.listdir(self.model_folder):
            if file.startswith(model_name):
                os.remove(os.path.join(self.model_folder, file))

    def load_recent(self, model_name:str='agentXXX'):
        files = [f for f in os.listdir(self.model_folder) if f.startswith(model_name)]
        if len(files) == 0:
            return
        # use file creation time as key
        files.sort(key=lambda x: os.path.getctime(self.model_folder + "/" + x))
        self.Q_table = np.load(self.model_folder + "/" + files[-1])
        return self.Q_table
    
    def export_model(self, folder_path):
        name = "model.npy"
        path = folder_path + "/" + name
        np.save(path, self.Q_table)