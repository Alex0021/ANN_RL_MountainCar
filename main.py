import gymnasium as gym
import numpy as np
from src.agents import RandomAgent
from src.stats import StatsRecorder
import matplotlib.pyplot as plt
import tqdm

#========
# GLOBALS
#========
MAX_EPISODES = 100
BULK_SIZE = 10

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

def generate_graphs(stats:StatsRecorder, agent_name:str="agent"):
    fig, axs = plt.subplots(3)
    plt.tight_layout(pad=3.0)
    fig.suptitle(f"Training stats for {agent_name}")

    bulk_rewards = stats.get_bulk_rewards()
    bulk_fraction_goal = stats.get_bulk_fraction_goal()
    bulk_episode_length = stats.get_bulk_episode_length()

    axs[0].plot(bulk_rewards[:,0], bulk_rewards[:,1], marker="o")
    axs[0].set_title("Rewards")
    axs[0].set_xlabel("Episodes")
    axs[0].set_ylabel("Rewards")

    axs[1].plot(bulk_fraction_goal[:,0], bulk_fraction_goal[:,1], marker="o")
    axs[1].set_title("Fraction of goals reached")
    axs[1].set_xlabel("Episodes")
    axs[1].set_ylabel("Fraction of goals reached")

    axs[2].scatter(bulk_episode_length[:,0], bulk_episode_length[:,1])
    axs[2].set_title("Episode length")
    axs[2].set_xlabel("Episodes")
    axs[2].set_ylabel("Episode length")

    plt.show()

def run_agent(env:gym.Env, agent:RandomAgent, stats:StatsRecorder):
    for i in tqdm.tqdm(range(stats.episode_num), ncols=75):
        done = False
        np.random.seed(np.random.randint(0, 1000))
        state, _ = env.reset()
        stats.start_recording()
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done  = terminated or truncated

            agent.observe(state, action, next_state, reward, done)

            state = next_state
            stats.record(reward)

        stats.stop_recording()


if __name__ == "__main__":
    # test()
    env = gym.make('MountainCar-v0')
    agent = RandomAgent(env.action_space.n, env.observation_space.shape[0], MAX_EPISODES=MAX_EPISODES)
    stats = StatsRecorder(MAX_EPISODES, 200, BULK_SIZE)
    run_agent(env, agent, stats)

    #replay_episode(agent.replay_buffer.buffer[0])

    generate_graphs(stats, agent_name=agent.__class__.__name__)

