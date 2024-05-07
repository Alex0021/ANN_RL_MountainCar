import gymnasium as gym
import numpy as np
from src.agents import RandomAgent, DqnAgent
from src.stats import StatsRecorder
import matplotlib.pyplot as plt
import tqdm
from src.ReplayBuffer import ReplayBuffer
import os
import sys

#========
# GLOBALS
#========

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

def replay_episode(replay_buffer:ReplayBuffer, episode:int):
    """
    Replays the stored episode in the environment.
    - gym: the gym environment
    - episode: the episode to replay nd array of (state, action, next_state, reward)
    """

    env = gym.make('MountainCar-v0', render_mode="human")
    env.reset()
    obs_dim = env.observation_space.shape[0]
    buffer = replay_buffer.buffer
    mapping = replay_buffer.mapping

    step = 0
    done = False
    while not done:
        step_array = buffer[mapping[(episode, step)]]
        action_index = obs_dim
        action = step_array[action_index]
        result = env.step(int(action))
        done = result[2] or result[3]
        print(f"Reward: {result[1]}, action: {action}")
        step += 1

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

def train_agent(env:gym.Env, agent:DqnAgent|RandomAgent, stats:StatsRecorder):
    EPISODE_DELAY = 5
    total_steps = 0
    total_episodes = 0
    for i in tqdm.tqdm(range(stats.episode_num), ncols=75):
        done = False
        np.random.seed(np.random.randint(0, 1000))
        state, _ = env.reset()
        stats.start_recording()
        while not done:
            action = agent.select_action(state, i)
            stats.record_action(int(action))
            next_state, reward, terminated, truncated, _ = env.step(int(action))
            done  = terminated or truncated
            agent.observe(state, action, next_state, reward, done)
            if total_steps >= agent.BATCH_SIZE:
                agent.update()
            stats.record_reward(reward)
            state = next_state
            total_steps += 1
        total_episodes += 1
        stats.stop_recording()

def evaluate_agent(path:str, MAX_EPISODES:int=1_000_000):
    stats = StatsRecorder(1_000_000, 
                        200, 
                        1,
                        3, 
                        log_dir="logs/eval")
    # env = gym.make('MountainCar-v0', render_mode="human")
    env = gym.make('gyms:gyms/CustomMountainCar-v0', render_mode="human")
    total_steps = 0
    total_episodes = 0
    agent = DqnAgent(env.action_space.n, env.observation_space.shape[0], eval=True)
    agent.load(path)
    while total_episodes < MAX_EPISODES:
        done = False
        state, _ = env.reset()
        stats.start_recording()
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(int(action))
            print(f"Reward: {reward}, Action: {action}", end="\r")
            env.render()
            env.plot_state(next_state, reward)
            done = terminated or truncated
            state = next_state
            total_steps += 1
            stats.record_reward(reward)
        stats.stop_recording()
        total_episodes += 1

def evaluate_last_model(MAX_EPISODES:int=1_000_000):
    folder = "models"
    
    stats = StatsRecorder(1_000_000, 
                        200, 
                        1,
                        3, 
                        log_dir="logs/eval")
    
    # env = gym.make('MountainCar-v0', render_mode="human")
    env = gym.make('gyms:gyms/CustomMountainCar-v0', render_mode="human")
    total_steps = 0
    total_episodes = 0
    agent = DqnAgent(env.action_space.n, env.observation_space.shape[0], eval=True)
    while total_episodes < MAX_EPISODES:
        files = os.listdir(folder)
        if len(files) == 0:
            raise ValueError("No models to evaluate")
        files.sort()
        latest_model = files[-1]
        path = os.path.join(folder, latest_model)
        agent.load(path)

        done = False
        state, _ = env.reset()
        stats.start_recording()
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(int(action))
            print(f"Reward: {reward}, Action: {action}", end="\r")
            env.render()
            env.plot_state(next_state, reward)
            done = terminated or truncated
            state = next_state
            total_steps += 1
            stats.record_reward(reward)
        stats.stop_recording()
        total_episodes += 1
    
if __name__ == "__main__":

    args = sys.argv[1:]

    if len(args) > 0:
        if args[0] == "--eval":
            evaluate_last_model()
            sys.exit(0)
        elif args[0] == "--train":
            # env = gym.make('MountainCar-v0')
            env = gym.make('gyms:gyms/CustomMountainCar-v0')

            # Training parameters
            total_episodes = 3000
            MAX_EPISODES = total_episodes // 30

            # agent parameters
            obs_dim = env.observation_space.shape[0]
            num_actions = env.action_space.n
            BULK_SIZE = 1
            MAX_STEPS = 200
            BATCH_SIZE = 512
            gamma = 0.99
            # epsilon = lambda iter: max(0.9*np.exp(-iter/(total_episodes/10)), 0.05)
            epsilon = lambda iter: max(0.9*np.exp(-(iter*10/total_episodes)), 0.07)
            alpha = 0.001

            stats = StatsRecorder(total_episodes, 
                                MAX_STEPS, 
                                BULK_SIZE,
                                num_actions,
                                log_dir="logs/train"
                                )
            
            agent = DqnAgent(num_actions, 
                            obs_dim, 
                            discount=gamma, 
                            epsilon=epsilon, 
                            alpha=alpha, 
                            MAX_STEPS=MAX_STEPS, 
                            MAX_EPISODES=MAX_EPISODES, 
                            BATCH_SIZE=BATCH_SIZE, 
                            stats=stats
                            )
            
            # agent = RandomAgent(env.action_space.n, env.observation_space.shape[0], MAX_EPISODES=MAX_EPISODES)
            train_agent(env, agent, stats)

            evaluate_last_model()

            # launch tensorboard
            os.system("./launch_tensorboard_train.sh")

            # replay_episode(agent.replay_buffer, MAX_EPISODES-1)

            # generate_graphs(stats, agent_name=agent.__class__.__name__)

            print("hello")
            sys.exit(0)
        else:
            raise ValueError("Invalid argument")

