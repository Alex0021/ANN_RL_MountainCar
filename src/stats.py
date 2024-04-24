import numpy as np

class StatsRecorder:
    def __init__(self, episode_num: int, steps_num: int, bulk_size: int):
        assert episode_num % bulk_size == 0, "Bulk size must be a multiple of the number of episodes."
        self.episode_num = episode_num
        self.steps_num = steps_num
        self.bulk_size = bulk_size
        self.episode_rewards = np.zeros((episode_num, steps_num), dtype=np.float32)
        self.episode_length = np.zeros((episode_num), dtype=np.int32)
        self.episode_index = 0
        self.step_index = 0

    def start_recording(self):
        assert self.episode_index < self.episode_num, "Number of episodes exceeded! Maybe call reset"
        self.step_index = 0
        
    def stop_recording(self):
        self.episode_length[self.episode_index] = self.step_index + 1
        self.episode_index += 1

    def reset(self):
        self.episode_rewards = np.zeros((self.episode_num, self.steps_num), dtype=np.float32)
        self.episode_length = np.zeros((self.episode_num), dtype=np.int32)
        self.episode_index = 0
        self.step_index = 0

    def record(self, reward: float):
        assert self.step_index < self.steps_num, "Number of steps exceeded! Maybe call to stop recording is missing"
        self.episode_rewards[self.episode_index, self.step_index] = reward
        self.step_index += 1

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

