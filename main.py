import gymnasium as gym
import numpy as np
from src.agents import RandomAgent, DqnAgent, DqnAgentRND, DynaAgent
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

def train_rnd_agent(env:gym.Env, agent:DqnAgentRND):
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

def train_agent(env:gym.Env, agent:DqnAgent|RandomAgent):
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
            next_state, tot_reward, terminated, truncated, infos = env.step(int(action))
            aux_reward = infos.get("aux_reward", 0)
            reward = infos.get("env_reward", 0)
            done  = terminated or truncated
            agent.observe(state, action, next_state, reward, aux_reward, done)
            agent.update()
            state = next_state
            total_steps += 1
        total_episodes += 1
        stats.stop_recording()

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

    args = sys.argv[1:]

    # env = gym.make('MountainCar-v0')
    env = gym.make('gyms:gyms/CustomMountainCar-v0')

    # Training parameters
    total_episodes = 3_000
    MAX_EPISODES = 500

    # agent parameters
    obs_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n
    BULK_SIZE = 1
    MAX_STEPS = 200
    BATCH_SIZE = 64
    gamma = 0.99
    epsilon = lambda iter: max(0.1*np.exp(-iter/(total_episodes/50)), 0.01)
    # epsilon = lambda iter: max(np.exp(-(iter*5/total_episodes)), 0.07)
    #epsilon = 0.1
    alpha = 1e-3
    reward_factor = lambda iter: np.exp(-(iter*10/(total_episodes*MAX_STEPS)))

    if len(args) < 3:
        raise ValueError("Needs at least 3 arguments: --agent [random|dqn|dqn-rnd] --train|--eval [--last|--model <model_number>]")

    if args[0] == "--train":
        eval_mode = False
    elif args[0] == "--eval":
        eval_mode = True
    else:
        raise ValueError("Specify train or eval mode as first argument")
    

    # Folder specified for eval
    if eval_mode:
        if args[1] == "--last":
            if "--agent" not in args:
                raise ValueError("No agent type specified")
            evaluate_last_model(agent_type=args[args.index("--agent")+1])
        elif args[1] == "--model":
            if len(args) > 2:
                agent_type = args[2].split("_")[0]
                path = os.path.join("models", f"{args[2]}")
                evaluate_agent(agent_type, path=path)
            else:
                raise ValueError("No model number provided")
        else:
            if "--agent" not in args:
                raise ValueError("No agent type specified")
            evaluate_last_model(agent_type=args[args.index("--agent")+1])
        sys.exit(0)
    
    
    stats = StatsRecorder(total_episodes, 
                        MAX_STEPS, 
                        BULK_SIZE,
                        num_actions,
                        log_dir="logs/train"
                        )
    
    agent_arg_idx = args.index("--agent")
    if args[agent_arg_idx+1] == "random":
        agent = RandomAgent(env.action_space.n, env.observation_space.shape[0], MAX_EPISODES=MAX_EPISODES)
    elif args[agent_arg_idx+1] == "dqn":
        agent = DqnAgent(env.action_space.n, 
                        env.observation_space.shape[0], 
                        discount=gamma, 
                        epsilon=epsilon, 
                        alpha=alpha, 
                        MAX_STEPS=MAX_STEPS, 
                        MAX_EPISODES=MAX_EPISODES, 
                        BATCH_SIZE=BATCH_SIZE, 
                        use_target_network=True,
                        eval=eval_mode,
                        stats=stats
                        )
    elif args[agent_arg_idx+1] == "dqn-rnd":
        agent = DqnAgentRND(env.action_space.n, 
                        env.observation_space.shape[0], 
                        discount=gamma, 
                        epsilon=epsilon, 
                        alpha=alpha, 
                        MAX_STEPS=MAX_STEPS, 
                        MAX_EPISODES=MAX_EPISODES, 
                        BATCH_SIZE=BATCH_SIZE, 
                        use_target_network=True,
                        eval=eval_mode,
                        stats=stats,
                        rnd_alpha=0.01,
                        reward_factor=reward_factor
        )
    elif args[agent_arg_idx+1] == "dyna":
        agent = DynaAgent(env.action_space.n, 
                        env.observation_space.shape[0], 
                        discount=gamma, 
                        epsilon=epsilon, 
                        MAX_STEPS=MAX_STEPS, 
                        MAX_EPISODES=MAX_EPISODES, 
                        eval=eval_mode,
                        stats=stats,
                        lr=0.2,
                        n_bins=(72,28),
                        k=BATCH_SIZE
        )
    else:
        raise ValueError("Invalid agent type")
    
    if agent.__class__.__name__ == "DqnAgentRND":
        train_rnd_agent(env, agent)
    else:
        train_agent(env, agent)

        #evaluate_last_model()

        # launch tensorboard
        #os.system("./launch_tensorboard_train.sh")

        # replay_episode(agent.replay_buffer, MAX_EPISODES-1)

        # generate_graphs(stats, agent_name=agent.__class__.__name__)

        print("hello")
        sys.exit(0)

