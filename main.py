import gymnasium as gym
import numpy as np
from src.agents import RandomAgent

def tutorial():
    env = gym.make('MountainCar-v0')
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # the second return value is an info dictionary, but it doesn't contain anything in this environment
    starting_state, _ = env.reset() 
    print(f"Starting state: {starting_state}")

    action = env.action_space.sample()
    print(f"Sampled action: {action}")
    next_state, reward, terminated, truncated, _ = env.step(action) # again, the last return value is an empty info object

    print(f"Next state: {next_state}")
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")

    done = False
    state, _ = env.reset()
    episode_reward = 0

    while not done:
        action = env.action_space.sample()
        next_state, reward, terimnated, truncated, _ = env.step(action)

        episode_reward += reward

        state = next_state
        done = terminated or truncated

    print(f"Episode reward after taking random actions: {episode_reward}")

def test():
    env = gym.make('MountainCar-v0', render_mode="human")
    starting_state, _ = env.reset() 
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, _ = env.step(action) # again, the last return value is an empty info object
    done = False
    state, _ = env.reset()
    episode_reward = 0

    while not done:
        action = env.action_space.sample()
        next_state, reward, terimnated, truncated, _ = env.step(action)

        episode_reward += reward

        state = next_state
        done = terminated or truncated

def replay_episode(episode:np.ndarray):
    """
    Replays the stored episode in the environment.
    - gym: the gym environment
    - episode: the episode to replay nd array of (state, action, next_state, reward)
    """

    env = gym.make('MountainCar-v0', render_mode="human")
    env.reset()
    env.state = episode[:2]

    for state_x, state_y, action, _, _, reward in episode:
        print(f"Reward: {reward}, action: {action}")
        env.step(int(action))
        env.render()

    

def run_agent(env:gym.Env, agent:RandomAgent):
    done = False
    state, _ = env.reset()
    episode_reward = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done  = terminated or truncated

        agent.observe(state, action, next_state, reward)

        state = next_state

if __name__ == "__main__":
    # test()
    env = gym.make('MountainCar-v0')
    agent = RandomAgent(env.action_space.n, env.observation_space.shape[0])
    run_agent(env, agent)

    replay_episode(agent.replay_buffer.buffer[0])

