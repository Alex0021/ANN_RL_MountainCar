from gymnasium.envs.registration import register

register(
     id="gyms/CustomMountainCar-v0",
     entry_point="gyms.envs:CustomMountainCar",
     max_episode_steps=200,
)