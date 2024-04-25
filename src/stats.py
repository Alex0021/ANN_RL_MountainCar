import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time

class StatsRecorder:
    def __init__(self, episode_num: int, steps_num: int, bulk_size: int, log_dir: str="logs/train"):
        assert episode_num % bulk_size == 0, "Bulk size must be a multiple of the number of episodes."
        self.episode_num = episode_num
        self.steps_num = steps_num
        self.bulk_size = bulk_size
        self.episode_rewards = np.zeros((episode_num, steps_num), dtype=np.float32)
        self.episode_length = np.zeros((episode_num), dtype=np.int32)
        self.episode_index = 0
        self.step_index = 0
        self.total_steps = 0
        dir_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.writer = SummaryWriter(log_dir=log_dir+"/"+dir_name, flush_secs=5, max_queue=5)
        # create and write "launch_tensorboard.sh" file
        path = log_dir + "/" + dir_name + "/launch_tensorboard.sh"
        with open(path, "w") as f:
            str = """
            # random port
            port=$(((RANDOM%1000)+6006))
            logdir="./logs/*"
            echo "Starting tensorboard: $logdir"
            open http://localhost:$port
            tensorboard --logdir=$logdir --port=$port
            """
            f.write(str)

    def start_recording(self):
        assert self.episode_index < self.episode_num, "Number of episodes exceeded! Maybe call reset"
        self.step_index = 0
        
    def stop_recording(self):
        self.episode_length[self.episode_index] = self.step_index + 1
        self.log(**{
            "episodes/episode_length": (self.episode_index, self.step_index + 1),
            "episodes/episode_reward": (self.episode_index, np.sum(self.episode_rewards[self.episode_index])),
        })
        self.writer.flush()
        self.episode_index += 1

    def reset(self):
        self.episode_rewards = np.zeros((self.episode_num, self.steps_num), dtype=np.float32)
        self.episode_length = np.zeros((self.episode_num), dtype=np.int32)
        self.episode_index = 0
        self.step_index = 0

    def record_reward(self, reward):
        assert self.step_index < self.steps_num, "Number of steps exceeded! Maybe call to stop recording is missing"
        self.episode_rewards[self.episode_index, self.step_index] = reward
        self.step_index += 1
        self.total_steps += 1
        self.log(**{
            "steps/step_reward": (self.total_steps, reward),
        })

    def log(self, **kwargs):
        for key, value in kwargs.items():
            self.writer.add_scalar(key, value[1], value[0])
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

