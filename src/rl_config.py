import numpy as np
import yaml

class RLConfig():
    def __init__(self, config:dict) -> None:
        default_config = yaml.safe_load(open("configs/default.yaml", "r"))
        for key, value in default_config.items():
            value = self.parse_value(value)
            setattr(self, key, value)
        # convert keys to class members
        for key, value in config.items():
            setattr(self, key, value)

        self.config = {**default_config, **config}

        if not isinstance(self.epsilon, float):
            self.epsilon_str = self.epsilon
            self.epsilon = lambda iter: eval(self.epsilon_str, {**self.config, "np": np }, {"iter":iter})

        # same for reward factor
        if not isinstance(self.reward_factor, float):
            self.reward_factor_str = self.reward_factor
            self.reward_factor = lambda iter: eval(self.reward_factor_str, {**self.config, "np": np }, {"iter":iter})

    def parse_value(self, value):
        try:
            new_value = eval(value)
            return new_value
        except:
            return value

    def export(self):
        yaml.dump(self.config, open(f"{self.data_path}/config.yaml", "w"))