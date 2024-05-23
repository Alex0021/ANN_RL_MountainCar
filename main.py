import gymnasium as gym
import numpy as np
from src.agents import RandomAgent, DqnAgent, DqnAgentRND, DynaAgent
from src.stats import StatsRecorder
import matplotlib.pyplot as plt
import tqdm
from src.ReplayBuffer import ReplayBuffer
import os
import sys
import yaml
from src.rl_config import RLConfig

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

def train_rnd_agent(env:gym.Env, agent:DqnAgentRND, config:RLConfig):
    total_steps = 0
    total_episodes = 0
    np.random.seed(np.random.randint(0, 10000000))
    stats = agent.stats
    # First pass: collect state data for mean and std calculation
    print("Collecting state data for RND normalization...")
    for i in range(agent.RND_NORMALIZE_DELAY):
        done = False
        state, _ = env.reset()
        while not done:
            action = np.random.randint(0, agent.num_actions)
            next_state, tot_reward, terminated, truncated, infos = env.step(action)
            done = terminated or truncated
            state = next_state
            total_steps += 1
    print("Starting training...")
    for i in tqdm.tqdm(range(stats.episode_num), ncols=75):
        done = False
        state, _ = env.reset()
        stats.start_recording()
        while not done:
            action = agent.select_action(state, i)
            next_state, tot_reward, terminated, truncated, infos = env.step(int(action))
            aux_reward = infos.get("aux_reward", 0)
            reward = infos.get("env_reward", 0)
            done = terminated or truncated
            agent.observe(state, action, next_state, reward, aux_reward, done)
            agent.update()
            state = next_state
            total_steps += 1
        total_episodes += 1
        stats.stop_recording()
    stats.export_data(path=config.data_path)
    config.export()

def train_agent(env:gym.Env, agent:DqnAgent|RandomAgent, config:RLConfig):
    total_steps = 0
    total_episodes = 0
    stats = agent.stats
    np.random.seed(np.random.randint(0, 10000000))
    for i in tqdm.tqdm(range(stats.episode_num), ncols=75):
        done = False
        state, _ = env.reset()
        stats.start_recording()
        while not done:
            action = agent.select_action(state, i)
            next_state, tot_reward, terminated, truncated, infos = env.step(int(action), reward_type=config.heuristic_reward_type)
            aux_reward = infos.get("aux_reward", 0)
            reward = infos.get("env_reward", 0)
            done  = terminated or truncated
            if not config.use_heuristic_reward:
                aux_reward = 0
            agent.observe(state, action, next_state, reward, aux_reward, done)
            agent.update()
            state = next_state
            total_steps += 1
        total_episodes += 1
        stats.stop_recording()
    stats.export_data(path=config.data_path)
    config.export()

def evaluate_agent(agent_type:str, path:str, MAX_EPISODES:int=1_000_000):
    stats = StatsRecorder(1_000_000, 
                        200, 
                        1,
                        3, 
                        log_dir="logs/eval")
    env = gym.make('gyms:gyms/CustomMountainCar-v0', render_mode="human")
    total_episodes = 0
    match agent_type:
        case DqnAgent.__class__.__name__:
            agent = DqnAgent(env.action_space.n, env.observation_space.shape[0], eval=True)
            path.join(path,".pth")
        case DqnAgentRND.__class__.__name__:
            agent = DqnAgentRND(env.action_space.n, env.observation_space.shape[0], eval=True)
            path.join(path,".pth")
        case DynaAgent.__class__.__name__:
            agent = DynaAgent(env.action_space.n, env.observation_space.shape[0], eval=True)
            path.join(path,".npy")
        case RandomAgent.__class__.__name__:
            agent = RandomAgent(env.action_space.n, env.observation_space.shape[0])
    print(f'Loading model: "{path}"')
    agent.load(path)
    while total_episodes < MAX_EPISODES:
        done = False
        state, _ = env.reset()
        stats.start_recording()
        episode_steps = 1
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, infos = env.step(int(action))
            aux_reward = infos.get("aux_reward", 0)
            reward = infos.get("env_reward", 0)
            print("                                            ", end="\r")
            print(f"Reward: {reward+aux_reward:.5f}, Action: {action}, Step: {episode_steps}", end="\r")
            env.render()
            env.plot_state(next_state, reward, agent=agent)
            done = terminated or truncated
            state = next_state
            episode_steps += 1
        stats.stop_recording()
        total_episodes += 1

def evaluate_last_model(agent_type:str, MAX_EPISODES:int=1_000_000):
    folder = "models"
    
    stats = StatsRecorder(1_000_000, 
                        200, 
                        1,
                        3, 
                        log_dir="logs/eval")
    
    # env = gym.make('MountainCar-v0', render_mode="human")
    env = gym.make('gyms:gyms/CustomMountainCar-v0', render_mode="human")
    total_episodes = 0
    match agent_type:
        case "dqn":
            agent = DqnAgent(env.action_space.n, env.observation_space.shape[0], eval=True)
        case "dqn-rnd":
            agent = DqnAgentRND(env.action_space.n, env.observation_space.shape[0], eval=True)
        case "dyna":
            agent = DynaAgent(env.action_space.n, env.observation_space.shape[0], n_bins=(72,28), eval=True)
        case "random":
            agent = RandomAgent(env.action_space.n, env.observation_space.shape[0])
    while total_episodes < MAX_EPISODES:
        files = os.listdir(folder)
        files = [f for f in files if f.startswith(agent.__class__.__name__)]
        if len(files) == 0:
            raise ValueError("No models to evaluate")
        files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        latest_model = files[-1]
        path = os.path.join(folder, latest_model)
        print(f'Loading model: "{path}"')
        agent.load(path)

        done = False
        state, _ = env.reset()
        stats.start_recording()
        episode_steps = 1
        while not done:
            action = agent.select_action(state)
            next_state, tot_reward, terminated, truncated, infos = env.step(int(action))
            reward, aux_reward = infos.get("env_reward", 0), infos.get("aux_reward", 0)
            print("                                            ", end="\r")
            print(f"Reward: {tot_reward:.5f}, Action: {action}, Step: {episode_steps}", end="\r")
            env.render()
            env.plot_state(next_state, reward, agent=agent)
            done = terminated or truncated
            state = next_state
            episode_steps += 1
        stats.stop_recording()
        total_episodes += 1

if __name__ == "__main__":

    config_path = sys.argv[1]
    config = yaml.safe_load(open(config_path, "r"))
    config = RLConfig(config)

    # env = gym.make('MountainCar-v0')
    env = gym.make(config.env)

    print(f"obs dim: {env.observation_space.shape[0]}, action dim: {env.action_space.n}")
    BULK_SIZE = 1

    if config.eval_mode:
        if config.eval_last_model:
            evaluate_last_model(agent_type=config.agent_type)
        else:
            evaluate_agent(config.agent_type, config.model_path)
        sys.exit(0)
    
    stats = StatsRecorder(config.num_episodes, 
                        config.max_steps, 
                        BULK_SIZE,
                        env.action_space.n,
                        log_dir=config.log_dir)
    
    if config.agent_type == "random":
        agent = RandomAgent(env.action_space.n, 
                            env.observation_space.shape[0], 
                            MAX_EPISODES=config.buffer_size, 
                            stats=stats)
    elif config.agent_type == "dqn":
        agent = DqnAgent(env.action_space.n, 
                        env.observation_space.shape[0], 
                        discount=config.gamma, 
                        epsilon=config.epsilon, 
                        alpha=config.alpha, 
                        MAX_STEPS=config.max_steps, 
                        MAX_EPISODES=config.buffer_size, 
                        BATCH_SIZE=config.batch_size, 
                        use_target_network=config.use_target_network,
                        eval=config.eval_mode,
                        stats=stats)
    elif config.agent_type == "dqn-rnd":
        agent = DqnAgentRND(env.action_space.n, 
                        env.observation_space.shape[0], 
                        discount=config.gamma, 
                        epsilon=config.epsilon, 
                        alpha=config.alpha, 
                        MAX_STEPS=config.max_steps, 
                        MAX_EPISODES=config.buffer_size, 
                        BATCH_SIZE=config.batch_size, 
                        use_target_network=config.use_target_network,
                        eval=config.eval_mode,
                        stats=stats,
                        rnd_alpha=config.rnd_alpha,
                        reward_factor=config.reward_factor,
        )
    elif config.agent_type == "dyna":
        agent = DynaAgent(env.action_space.n, 
                        env.observation_space.shape[0], 
                        discount=config.gamma, 
                        epsilon=config.epsilon, 
                        MAX_STEPS=config.max_steps, 
                        MAX_EPISODES=config.buffer_size, 
                        eval=config.eval_mode,
                        stats=stats,
                        lr=config.alpha,
                        n_bins=config.n_bins,
                        k=config.k,
        )
    else:
        raise ValueError("Invalid agent type")
    
    if agent.__class__.__name__ == "DqnAgentRND":
        train_rnd_agent(env, agent, config)
    else:
        train_agent(env, agent, config)

        #evaluate_last_model()

        # launch tensorboard
        #os.system("./launch_tensorboard_train.sh")

        # replay_episode(agent.replay_buffer, MAX_EPISODES-1)

        # generate_graphs(stats, agent_name=agent.__class__.__name__)

        print("hello")
        sys.exit(0)

