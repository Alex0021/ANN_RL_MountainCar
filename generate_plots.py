import os, sys
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
import yaml
import pandas as pd

from src.rl_config import RLConfig
from src.data_utils import extract_data_from_event, extract_data_from_tb_file

from src.agents import DynaAgent
from main import evaluate_agent

def configure_matplotlib():
    # print(plt.style.available)
    plt.style.use('seaborn-v0_8-pastel')
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 12
    plt.rcParams['text.usetex'] = True

def smooth_data(data, window_size=5):
    data_smoothed = np.zeros_like(data)
    for i in range(window_size-1, len(data)):
        data_smoothed[i] = np.mean(data[max(0, i-window_size):i+1])
    for i in range(window_size-1):
        data_smoothed[i] = np.mean(data[:i+1])
    return data_smoothed

def plot_loss(df, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Loss")

    metric = df[df.metric == "training/loss"]
    ax.scatter(metric.step[::10], metric.value[::10], label="Loss", s=1)
    value = smooth_data(metric.value, window_size=100)
    ax.plot(metric.step, value, label="Smoothed", color='orange')

def plot_loss_avg_window(data_dict, window_size=5_000):
    if 'loss' not in data_dict:
        print("No loss data found in dictionary")
        return
    print("Plotting loss over average window")
    data = np.array(data_dict['loss'])
    update_steps = [d[0] for d in data]
    loss_values = np.array([d[1] for d in data])
    loss_values_avg = np.zeros_like(loss_values)
    for i in range(window_size-1, len(loss_values)):
        loss_values_avg[i] = np.mean(loss_values[max(0, i-window_size):i+1])
    plt.figure()
    plt.plot(update_steps[window_size-1:], loss_values_avg[window_size-1:])
    plt.title(f"Loss (MSE) over an average window of {window_size} steps")
    plt.xlabel("Update steps")
    plt.ylabel("Loss")

def plot_episode_length(data_dict):
    if 'episode_length' not in data_dict:
        print("No episode length data found in dictionary")
        return
    print("Plotting episode length")
    data = np.array(data_dict['episode_length'])
    episode_steps = [d[0] for d in data]
    episode_length = [d[1] for d in data]
    plt.figure()
    plt.scatter(episode_steps, episode_length)
    plt.title("Episode duration")
    plt.xlabel("Episodes")
    plt.ylabel("Episode length")

def plot_episode_rewards(data_dict, window_size=100, ax=None):
    if 'episode_reward' not in data_dict:
        print("No episode reward data found in dictionary")
        return
    print("Plotting episode reward")
    data = np.array(data_dict['episode_reward'])
    episode_steps = [d[0] for d in data]
    episode_reward = [d[1] for d in data]
    episode_reward_avg = np.zeros_like(episode_reward)
    for i in range(window_size-1, len(episode_reward)):
        episode_reward_avg[i] = np.mean(episode_reward[max(0, i-window_size):i+1])
    
    plt.figure()
    plt.plot(episode_steps[window_size-1:], episode_reward_avg[window_size-1:], linewidth=1.5)
    plt.title(f"Episode reward over an average window of {window_size} episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Episode reward")

def plot_cumulative_success(df):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success count")
    ax.set_title("Cumulative success count")

    metric = df[df.metric == "episodes/episode_completed"]
    ax.plot(metric.step, np.cumsum(metric.value), label="No heuristic")

def parse_args_for_tb():
    args = sys.argv[1:]
    if len(args) > 0:
        if args[0] == "--folder":
            data_dict = {}
            folder = args[1]
            try:
                files = os.listdir(folder)
                path = None
                for f in files:
                    if 'events.out.tfevents' in f:
                        path = os.path.join(folder, f)
                        break
                if path is None:
                    print("No event file found")
                    sys.exit(1)
                print(f"Opening {path}")
                # Extracting data from the event file
                data_dict = extract_data_from_tb_file(path)
                # Generating plots
                plot_loss(data_dict)
                plot_loss_avg_window(data_dict)
                plot_cumulative_success(data_dict)
                print("DONE")
                plt.show()
            except FileNotFoundError:
                print("Folder not found")
                sys.exit(1)

def get_data(plot_name, data_file="data.csv"):
    dir = "./plot_data/" + plot_name 
    dirs = os.listdir(dir)
    data = {}
    for d in dirs:
        if d.startswith("."):
            continue
        data[d] = {
            "data": pd.read_csv(os.path.join(dir, d)+"/"+data_file),
            "config": RLConfig(yaml.load(open(os.path.join(dir, d)+"/config.yaml"), Loader=yaml.FullLoader))
        }
    return data

def plot_episode_length(df, window_size=100):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Length")
    ax.set_title("Episode length")

    metric = df[df.metric == "episodes/episode_length"]
    ax.scatter(metric.step, metric.value, s=1)

    ax.plot(metric.step, smooth_data(metric.value, window_size=100), label="Smoothed", color='orange')

def plot_cumulative_reward_per_episode(df):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative reward")
    ax.set_title("Cumulative reward per episode")

    metric = df[df.metric == "episodes/episode_reward"]
    ax.scatter(metric.step, metric.value, label="No heuristic", s=1)
    ax.plot(metric.step, smooth_data(metric.value, window_size=100), label="Smoothed", color='orange')

def plot_cumulative_reward_by_type(df, episode_limit=500):
    df = df[df.step <= episode_limit]

    env_reward = df[df.metric == "episodes/episode_env_reward"]
    aux_reward = df[df.metric == "episodes/episode_aux_reward"]
    tot_reward = df[df.metric == "episodes/episode_reward"]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Cumulative reward by type")

    ax.plot(env_reward.step, np.cumsum(env_reward.value), label="Environment")
    ax.plot(aux_reward.step, np.cumsum(aux_reward.value), label="Heuristic")
    ax.plot(tot_reward.step, np.cumsum(tot_reward.value), label="Total")

    ax.legend()

# PLOT #1 (2)
def random_agent_plot():
    data = get_data("random_agent")
    df = data["random_agent"]["data"]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.set_yticks([200])

    ax.set_title("Episode length")

    metric = df[df.metric == "episodes/episode_length"]
    ax.plot(metric.step, metric.value, label="Episode length")

# PLOT #2 (3.2)
def dqn_no_heuristic_reward_loss():
    data = get_data("dqn_agent")
    df = data["no_heuristic_no_target"]["data"]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")

    metric = df[df.metric == "training/loss"]
    # filter outliers
    value = smooth_data(metric.value, window_size=100)
    step = metric.step
    metric = metric[metric.value < 4_000]
    
    ax.scatter(metric.step[::10], metric.value[::10], label="Loss", s=1)
    ax.plot(step, value, label="Smoothed", color='orange')

    # df = data["heuristic"]["data"]
    # metric = df[df.metric == "episodes/episode_reward"]
    # ax.scatter(metric.step, metric.value, label="Normed speed heuristic")

def dqn_no_heuristic_reward_episode_reward():
    data = get_data("dqn_agent")
    df = data["no_heuristic_no_target"]["data"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative reward")
    ax.set_title("Cumulative reward per episode")

    metric = df[df.metric == "episodes/episode_reward"]
    ax.scatter(metric.step, metric.value, label="No heuristic", s=1)
    ax.plot(metric.step, smooth_data(metric.value, window_size=100), label="Smoothed", color='orange')


# PLOT #3 (3.3)
def dqn_heuristic_episode_length():
    data = get_data("dqn_agent")
    df = data["heuristic"]["data"]

    plot_episode_length(df)

# PLOT #4 (3.3)
def dqn_heuristic_cumulative_reward_per_episode():
    data = get_data("dqn_agent")
    df = data["heuristic"]["data"]
    
    plot_cumulative_reward_per_episode(df)

def dqn_heuristic_cumulative_reward_by_type():
    data = get_data("dqn_agent")
    df = data["heuristic"]["data"]
    
    plot_cumulative_reward_by_type(df, episode_limit=500)

# PLOT #5 (3.3)
def dqn_heuristic_cumulative_success():
    data = get_data("dqn_agent")
    df = data["heuristic"]["data"]

    plot_cumulative_success(df)

# PLOT #6 (3.3)
def dqn_heuristic_loss():
    data = get_data("dqn_agent")
    df = data["heuristic"]["data"]

    plot_loss(df)

# PLOT #7 (3.4)
def dqn_rnd_episode_length():
    data = get_data("dqn_agent")
    df = data["rnd"]["data"]

    plot_episode_length(df)

# PLOT #8 (3.4)
def dqn_rnd_cumulative_reward_per_episode():
    data = get_data("dqn_agent")
    df = data["rnd"]["data"]
    
    plot_cumulative_reward_per_episode(df)

def dqn_rnd_cumulative_reward_by_type():
    data = get_data("dqn_agent")
    df = data["rnd"]["data"]
    
    plot_cumulative_reward_by_type(df)

# PLOT #9 (3.4)
def dqn_rnd_cumulative_success():
    data = get_data("dqn_agent")
    df = data["rnd"]["data"]

    plot_cumulative_success(df)

# PLOT #10 (3.4)
def dqn_rnd_loss():
    data = get_data("dqn_agent")
    df = data["rnd"]["data"]

    plot_loss(df)

# PLOT #11 (4.4)
def dyna_episode_length():
    data = get_data("dyna_agent")
    df = data["dyna"]["data"]

    plot_episode_length(df)

# PLOT #12 (4.4)
def dyna_cumulative_reward_per_episode():
    data = get_data("dyna_agent")
    df = data["dyna"]["data"]
    
    plot_cumulative_reward_per_episode(df)

# PLOT #13 (4.4)
def dyna_cumulative_success():
    data = get_data("dyna_agent")
    df = data["dyna"]["data"]
    
    plot_cumulative_success(df)

# PLOT #14 (4.4)
def dyna_loss():
    data = get_data("dyna_agent")
    df = data["dyna"]["data"]

    loss = df[df.metric == "training/dQ"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Training steps")
    ax.set_ylabel("$\Delta Q$")
    ax.set_title("$\Delta Q$ over training steps")

    ax.set_ylim(-0.3, 0.3)

    value = smooth_data(loss.value, window_size=1000)
    ax.plot(loss.step, value, label="Smoothed", color='orange')
    ax.scatter(loss.step[::10], loss.value[::10], s=1)

def dyna_start_pos():
    data = get_data("dyna_agent", data_file="eval_data.csv")
    df = data["dyna"]["data"]

    start_pos = df[df.metric == "states/start_pos"]
    ep_len = df[df.metric == "episodes/episode_length"]
    fig = plt.figure()  
    ax = fig.add_subplot(111)
    ax.set_xlabel("Starting position")
    ax.set_ylabel("Episode length")
    ax.set_title("Episode length vs starting position")
    ax.scatter(start_pos.value, ep_len.value)

    ax.vlines(x=-0.5236, ymin=np.min(ep_len.value), ymax=np.max(ep_len.value), color='r', linestyle='--', linewidth=2)


# PLOT #15 (4.4)
def dyna_Q_values():
    agent = DynaAgent(3,2)
    agent.load("./plot_data/dyna_agent/dyna/model.npy")
    config = RLConfig(yaml.load(open("./plot_data/dyna_agent/dyna/config.yaml"), Loader=yaml.FullLoader))

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")

    ax.set_title("Q-values")

    Q_table = agent.Q_table

    max_Q = np.max(Q_table, axis=1)
    new_Q_table = max_Q.reshape((config.n_bins[0]+1, config.n_bins[1]+1))

    masked_Q_table = np.ma.masked_where(new_Q_table == 0, new_Q_table)

    # ax.imshow(new_Q_table, cmap='coolwarm', aspect='auto', extent=[-1.2, 0.6, -0.07, 0.07])
    ax.imshow(masked_Q_table, cmap='coolwarm', aspect='auto', extent=[-1.2, 0.6, -0.07, 0.07], origin='lower')

# PLOT #16 (4.4)
def dyna_key_episodes():
    key_episode_index = [9, 99, 999, 2999]
    models = [1000, 10_000, 100_000, 374_000]

    config = RLConfig(yaml.load(open("./plot_data/dyna_agent/dyna/config.yaml"), Loader=yaml.FullLoader))

    data = np.load("./plot_data/dyna_agent/dyna/traces.npy", allow_pickle=True)
    fig = plt.figure(figsize=(10, 10))

    # for Q values
    # agent = DynaAgent(3,2)
    # config = RLConfig(yaml.load(open("./plot_data/dyna_agent/dyna/config.yaml"), Loader=yaml.FullLoader))
    # agent.load(f"./plot_data/dyna_agent/dyna/model.npy")
    
    # colors = plt.get_cmap('viridis')(np.linspace(0, 1, 200))

    def split_data(data):
        diff = np.diff(data, axis=0)
        diff = np.abs(diff)
        mean_diff = np.mean(diff, axis=0)
        split = np.where(diff > mean_diff*10 ) 
        return np.split(data, split[0]+1) 

    for i in range(len(key_episode_index)):

        episode = key_episode_index[i]
        model = models[i]
        Q_table = np.load(f"./models/DynaAgent_model_{model}.npy")
        R_table = np.load(f"./models/DynaAgent_model_{model}_R_table.npy")

        ax = fig.add_subplot(len(key_episode_index)//2, len(key_episode_index)//2, i+1)
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        
        # ax.set_xlim(-1.5, 1)
        # ax.set_ylim(-0.1, 0.1) 
        # ax.set_xticks([-1.2, -0.6, 0, 0.6])
        # ax.set_yticks([-0.07, 0, 0.07])

        # Q values
        max_Q = np.max(Q_table, axis=1)
        R_table = np.sum(abs(R_table), axis=1)
        max_Q = np.ma.masked_where(R_table == 0, max_Q)  # filter out zero R
        new_Q_table = max_Q.reshape((config.n_bins[0]+1, config.n_bins[1]+1))
        ax.imshow(new_Q_table, cmap='coolwarm', aspect='auto', extent=[-1.2, 0.6, -0.07, 0.07], origin='lower')
        # ax.imshow(masked_Q_table, cmap='coolwarm', aspect='auto', origin='lower')
        
        # trajectory
        episode_data = data[episode]
        # filter out zeros
        episode_data = episode_data[~np.all(episode_data == 0, axis=1)]
        n = episode_data.shape[0]

        ax.set_title(f"Episode {episode+1}, length: {n} steps")

        # bins = np.linspace(-1.2, 0.6, config.n_bins[0]+1)
        # episode_data[:, 0] = np.digitize(episode_data[:, 0], bins=bins) - 1
        # bins = np.linspace(-0.07, 0.07, config.n_bins[1]+1)
        # episode_data[:, 1] = np.digitize(episode_data[:, 1], bins=bins) - 1

        # ax.scatter(episode_data[:,0], episode_data[:,1], color=colors[:n])
        for split in split_data(episode_data):
            ax.plot(split[:,0], split[:,1], c="black")

        # vertical line at 0.5
        ax.axvline(x=0.5, color='r', linestyle='--', linewidth=2)

    plt.tight_layout(
        rect=[0, 0.03, 1, 0.95]
    )




def dyna_trace_animation():
    data = np.load("./plot_data/dyna_agent/dyna/traces.npy", allow_pickle=True)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_title("Episode trace")


    colors = plt.get_cmap('viridis')(np.linspace(0, 1, 200))

    for i, episode in enumerate(data):
        episode = episode[~np.all(episode == 0, axis=1)]
        n = episode.shape[0]
        ax.set_xlim(-1.5, 1)
        ax.set_ylim(-0.1, 0.1)
        ax.set_xticks([-1.2, -0.6, 0, 0.6])
        ax.set_yticks([-0.07, 0, 0.07])
        ax.scatter(episode[:,0], episode[:,1], color=colors[:n])
        plt.savefig(f"./plot_data/dyna_agent/dyna/frames/frame_{i:04d}.png")
        ax.cla()


# PLOT #17 (4.4) (bonus)
def dyna_Q_values_at_key_episodes():
    models = [1000, 10_000, 50_000, 384_000]
    agent = DynaAgent(3,2)
    config = RLConfig(yaml.load(open("./plot_data/dyna_agent/dyna/config.yaml"), Loader=yaml.FullLoader))

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(r"$Q$-values at key episodes")

    for i, model in enumerate(models):
        agent.load(f"./models/DynaAgent_model_{model}.npy")
        ax = fig.add_subplot(len(models)//2, len(models)//2, i+1)
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")

        ax.set_title(f"{model} updates")

        Q_table = agent.Q_table

        max_Q = np.max(Q_table, axis=1)
        new_Q_table = max_Q.reshape((config.n_bins[0]+1, config.n_bins[1]+1))
        masked_Q_table = np.ma.masked_where(new_Q_table == -1, new_Q_table)
        # ax.imshow(new_Q_table, cmap='coolwarm', aspect='auto', extent=[-1.2, 0.6, -0.07, 0.07])
        ax.imshow(masked_Q_table, cmap='coolwarm', aspect='auto', extent=[-1.2, 0.6, -0.07, 0.07], origin='lower')


# PLOT #18 (4.5) 
def comparison_env_rewards():
    
    # load the data
    data = get_data("dqn_agent")
    data = [
        data["no_heuristic"],
        data["heuristic"],
        data["rnd"],
    ]

    data.append(get_data("dyna_agent")["dyna"])

    data = [d["data"]for d in data]

    metrics = [d[d.metric == "episodes/episode_env_reward"] for d in data]
    
    fig = plt.figure(figsize=(12, 6))
    
    fig.suptitle("Training performance")

    ax = fig.add_subplot(111)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Environment Reward")

    names = [
            "No heuristic", 
            "Heuristic",
            "RND",
            "Dyna"
    ]

    for name, metric in zip(names, metrics):
        value = smooth_data(metric.value, window_size=100)
        ax.plot(metric.step, value, label=name)

    ax.legend(loc="upper left")


# PLOT #19 (4.5)
def comparison_eval_performance():
    seeds = [np.random.randint(0, 100000) for _ in range(1000)]

    # load the models
    paths = [
        "dqn_agent/no_heuristic",
        "dqn_agent/heuristic",
        "dqn_agent/rnd",
        "dyna_agent/dyna",
    ]

    configs = [
        RLConfig(yaml.load(open("./plot_data/"+path+"/config.yaml"), Loader=yaml.FullLoader))
        for path in paths
    ]

    for config in configs:
        config.num_episodes = 1000
        config.eval_mode = True

    # run the models
    results = []
    for config in configs:
        evaluate_agent(config, seeds, render=False)

    paths = [
        "dqn_agent/no_heuristic",
        "dqn_agent/heuristic",
        "dqn_agent/rnd",
        "dyna_agent/dyna",
    ]

    # load the results
    for path in paths:
        results.append(pd.read_csv("./plot_data/"+path+"/eval_data.csv"))

    # plot the results
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle("Evaluation performance")
    ax = fig.add_subplot(111)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Environment Reward")

    names = [
        "No heuristic", 
        "Heuristic", 
        "RND", 
        "Dyna"
    ]

    for name, results in zip(names, results):
        metric = results[results.metric == "episodes/episode_env_reward"]
        value = smooth_data(metric.value, window_size=100)
        ax.plot(metric.step, value, label=name)

    ax.legend(loc='upper right')


# EXTRA (compare heuristic)
def dqn_compare_heuristic_rewards():
    data = get_data("dqn_agent")
    data = [
        data["heuristic_speed"],
        data["heuristic_height"],
        data["heuristic_speed_height"],
        data["heuristic_max_height"]
    ]

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode length")

    labels = ["Normed speed", "Normed height", "Normed speed \& height", "Maximum height"]

    for d, l in zip(data, labels):
        metric = d["data"][d["data"].metric == "episodes/episode_length"]
        value = smooth_data(metric.value, window_size=100)
        ax.plot(metric.step, value, label=l)
    
    ax.legend()


def target_network_effect_no_heuristic():
    data = get_data("dqn_agent")

    no_heuristic = [
        data["no_heuristic_no_target"],
        data["no_heuristic_target"],
    ]

    fig = plt.figure()
    # fig.suptitle("Effect of target network on DQN performance")

    labels = ["No target", "Target"]

    ax = fig.add_subplot(111)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode length")
    ax.set_title("With heuristic reward")
    for d, l in zip(no_heuristic, labels):
        metric = d["data"][d["data"].metric == "episodes/episode_length"]
        value = smooth_data(metric.value, window_size=100)
        ax.plot(metric.step, value, label=l)

    ax.legend(loc='upper right')
    
def target_network_effect_heuristic():
    data = get_data("dqn_agent")
    
    heuristic = [
        data["heuristic_no_target"],
        data["heuristic"],
    ]

    labels = ["No target", "Target"]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode length")
    ax.set_title("Without heuristic reward")
    for d, l in zip(heuristic, labels):
        metric = d["data"][d["data"].metric == "episodes/episode_length"]
        value = smooth_data(metric.value, window_size=100)
        ax.plot(metric.step, value, label=l)
    
    ax.legend(loc='upper right')


def generate_all_plots():
    configure_matplotlib()
    fs = [
        # random_agent_plot,
        # dqn_no_heuristic_reward_episode_reward,
        # dqn_no_heuristic_reward_loss,
        # dqn_heuristic_episode_length,
        # dqn_heuristic_cumulative_reward_per_episode,
        # dqn_heuristic_cumulative_reward_by_type,
        # dqn_heuristic_cumulative_success,
        # dqn_heuristic_loss,
        # dqn_rnd_episode_length,
        # dqn_rnd_cumulative_reward_by_type,
        # dqn_rnd_cumulative_reward_per_episode,
        # dqn_rnd_cumulative_success,
        # dqn_rnd_loss,
        # dyna_episode_length,
        # dyna_cumulative_reward_per_episode,
        # dyna_cumulative_success,
        # dyna_start_pos,
        # dyna_loss,
        # dyna_Q_values,
        dyna_key_episodes,
        # dyna_Q_values_at_key_episodes,
        # comparison_env_rewards,
        # comparison_eval_performance,

        # extra
        # dqn_compare_heuristic_rewards,
        # target_network_effect_heuristic,
        # target_network_effect_no_heuristic
    ]

    for f in fs:
        f()
        plt.savefig(f"./plots/{f.__name__}.svg")
        plt.close()



if __name__ == "__main__":
    configure_matplotlib()
    # random_agent_plot()
    # dqn_no_heuristic_reward()
    # dqn_heuristic_episode_length()
    # dqn_heuristic_cumulative_reward()
    # dqn_heuristic_cumulative_success()
    # dqn_heuristic_loss()
    # dqn_rnd_episode_length()
    # dqn_rnd_cumulative_reward()
    # dqn_rnd_cumulative_success()
    # dqn_rnd_loss()
    # dyna_episode_length()
    # dyna_cumulative_reward()
    # dyna_cumulative_success()
    # dyna_start_pos()
    # dyna_loss()
    # dyna_Q_values()
    # dyna_key_episodes()
    # dyna_trace_animation()
    # dyna_Q_values_at_key_episodes()
    # comparison_env_rewards()
    # comparison_eval_performance()
    # dqn_compare_heuristic_rewards()
    # target_network_effect()

    generate_all_plots()

    plt.show()
    