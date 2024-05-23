import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
import platform
from src.data_utils import tflog2pandas
import os

SH_SCRIPT = """
            # Check most recent dir
            logdir="./logs/*"
            # Check env var
            SERVER_MODE=${TENSORBOARD_SERVER:-0}
            if [ $SERVER_MODE -eq 1 ]; then
                # fixed port
                port=6006
                echo "Starting tensorboard: $logdir"
                tensorboard --logdir=$logdir --port=$port --bind_all
            else
                # random port
                port=$(((RANDOM%1000)+6006))
                echo "Starting tensorboard: $logdir"
                open http://localhost:$port
                tensorboard --logdir=$logdir --port=$port
            fi
            """

BAT_SCRIPT = """
            @echo off
            rem # random port
            set /a "port=(%RANDOM% %% 1000) + 6006"
            set logdir=./logs/*
            echo Starting tensorboard: %logdir%
            start http://localhost:%port%
            tensorboard --logdir=%logdir% --port=%port%
            """

class StatsRecorder:
    def __init__(self, episode_num: int, steps_num: int, bulk_size: int, action_num:int, log_dir: str="logs/train"):
        assert episode_num % bulk_size == 0, "Bulk size must be a multiple of the number of episodes."
        self.episode_num = episode_num
        self.steps_num = steps_num
        self.bulk_size = bulk_size
        self.episode_rewards = np.zeros((episode_num, steps_num), dtype=np.float32)
        self.env_rewards = np.zeros((episode_num, steps_num), dtype=np.float32)
        self.aux_rewards = np.zeros((episode_num, steps_num), dtype=np.float32)
        self.episode_length = np.zeros((episode_num), dtype=np.int32)
        self.episode_index = 0
        self.step_index = 0
        self.total_steps = 0
        self.actions_count = np.zeros(action_num, dtype=np.int32)
        dir_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.writer = SummaryWriter(log_dir=log_dir+"/"+dir_name, flush_secs=5, max_queue=5)
        self.current_epsilon = 0.0
        # create and write "launch_tensorboard" file
        if platform.system() == "Windows":
            file_extension = ".bat"
            str_code = BAT_SCRIPT
        else:
            file_extension = ".sh"
            str_code = SH_SCRIPT
        path = log_dir + "/" + dir_name + "/launch_tensorboard" + file_extension
        with open(path, "w") as f:
            f.write(str_code)

    def start_recording(self):
        assert self.episode_index < self.episode_num, "Number of episodes exceeded! Maybe call reset"
        self.step_index = 0
        
    def stop_recording(self):
        self.episode_length[self.episode_index] = self.step_index
        self.log(**{
            "episodes/episode_length": (self.episode_index, self.step_index),
            "episodes/episode_reward": (self.episode_index, np.sum(self.episode_rewards[self.episode_index])),
            "episodes/episode_completed": (self.episode_index, self.step_index < self.steps_num - 1),
            "episodes/episode_actions": (self.episode_index, {"left":self.actions_count[0],
                                                               "stay":self.actions_count[1],
                                                               "right":self.actions_count[2]}),
            "episodes/episode_epsilon": (self.episode_index, self.current_epsilon),
            "episodes/episode_env_reward": (self.episode_index, np.sum(self.env_rewards[self.episode_index])),
            "episodes/episode_aux_reward": (self.episode_index, np.sum(self.aux_rewards[self.episode_index]))
        })
        self.writer.flush()
        self.episode_index += 1
        self.actions_count = np.zeros_like(self.actions_count)


    def reset(self):
        self.episode_rewards = np.zeros((self.episode_num, self.steps_num), dtype=np.float32)
        self.env_rewards = np.zeros((self.episode_num, self.steps_num), dtype=np.float32)
        self.aux_rewards = np.zeros((self.episode_num, self.steps_num), dtype=np.float32)
        
        self.episode_length = np.zeros((self.episode_num), dtype=np.int32)

        self.episode_index = 0
        self.step_index = 0

    def record_reward(self, reward ,aux_reward):
        assert self.step_index < self.steps_num, "Number of steps exceeded! Maybe call to stop recording is missing"
        self.episode_rewards[self.episode_index, self.step_index] = reward+aux_reward
        self.aux_rewards[self.episode_index, self.step_index] = aux_reward
        self.env_rewards[self.episode_index, self.step_index] = reward

        self.step_index += 1
        self.total_steps += 1
        self.log(**{
            "steps/env_reward": (self.total_steps, reward),
            "steps/aux_reward": (self.total_steps, aux_reward),
            "steps/total_reward": (self.total_steps, reward+aux_reward)
        })

    def record_action(self, action):
        self.actions_count[action] += 1
        self.log(**{
            "steps/step_action": (self.total_steps, action)
        })

    def record_epsilon(self, epsilon):
        self.current_epsilon = epsilon

    def log(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value[1], dict):
                self.writer.add_scalars(key, value[1], value[0])
            else:
                self.writer.add_scalar(key, value[1], value[0])
        self.writer.flush()

    def log_histogram(self, **kwargs):
        for key, value in kwargs.items():
            self.writer.add_histogram(key, value[1], value[0])
        self.writer.flush()

    def get_bulk_rewards(self):
        multiple = self.episode_num // self.bulk_size
        points = np.zeros((multiple, 2))
        for i in range(multiple):
            points[i,:] = np.array([(i + 1)*self.bulk_size, np.mean(np.sum(self.episode_rewards[i*self.bulk_size : (i + 1)*self.bulk_size], axis=1), axis=0)])
        return points
    
    def get_bulk_fraction_goal(self):
        multiple = self.episode_num // self.bulk_size
        points = np.zeros((multiple, 2))
        for i in range(multiple):
            points[i,:] = np.array([(i + 1)*self.bulk_size, np.count_nonzero(self.episode_length[i*self.bulk_size : (i + 1)*self.bulk_size] < self.steps_num)/self.bulk_size])
        return points

    def get_bulk_episode_length(self):
        multiple = self.episode_num // self.bulk_size
        points = np.zeros((multiple, 2))
        for i in range(multiple):
            points[i,:] = np.array([(i + 1)*self.bulk_size,np.mean(self.episode_length[i*self.bulk_size : (i + 1)*self.bulk_size])])
        return points

    def export_data(self, path):
        df = tflog2pandas(self.writer.log_dir)
        if not os.path.exists(path):
            os.makedirs(path)
        path = path + "/data.csv"
        df.to_csv(path)
        

        



