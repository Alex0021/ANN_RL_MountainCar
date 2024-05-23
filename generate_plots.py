import os, sys
import matplotlib.pyplot as plt
import numpy as np
import time
import yaml
import pandas as pd

from src.rl_config import RLConfig
from src.data_utils import extract_data_from_event, extract_data_from_tb_file

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

def plot_loss(df):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")

    metric = df[df.metric == "training/loss"]
    ax.plot(metric.step, metric.value, label="Loss")

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

def plot_episode_rewards(data_dict, window_size=100):
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

def get_data(plot_name):
    dir = "./plot_data/" + plot_name 
    dirs = os.listdir(dir)
    data = {}
    for d in dirs:
        data[d] = {
            "data": pd.read_csv(os.path.join(dir, d)+"/data.csv"),
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

def plot_cumulative_reward(df):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative reward")
    ax.set_title("Cumulative reward per episode")

    metric = df[df.metric == "episodes/episode_reward"]
    ax.scatter(metric.step, metric.value, label="No heuristic", s=1)
    ax.plot(metric.step, smooth_data(metric.value, window_size=100), label="Smoothed", color='orange')

    env_reward = df[df.metric == "episodes/episode_env_reward"]
    aux_reward = df[df.metric == "episodes/episode_aux_reward"]
    tot_reward = df[df.metric == "episodes/episode_reward"]

    ax = fig.add_subplot(122)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Cumulative reward")

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
def dqn_no_heuristic_reward():
    data = get_data("dqn_agent")
    df = data["no_heuristic"]["data"]

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121)
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")

    metric = df[df.metric == "training/loss"]
    ax.plot(metric.step, metric.value, label="Loss")

    ax = fig.add_subplot(122)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative reward")
    ax.set_title("Cumulative reward per episode")

    metric = df[df.metric == "episodes/episode_reward"]
    ax.scatter(metric.step, metric.value, label="No heuristic")

    # df = data["heuristic"]["data"]
    # metric = df[df.metric == "episodes/episode_reward"]
    # ax.scatter(metric.step, metric.value, label="Normed speed heuristic")

    plt.show()

# PLOT #3 (3.3)
def dqn_heuristic_episode_length():
    data = get_data("dqn_agent")
    df = data["heuristic"]["data"]

    plot_episode_length(df)

# PLOT #4 (3.3)
def dqn_heuristic_cumulative_reward():
    data = get_data("dqn_agent")
    df = data["heuristic"]["data"]
    
    plot_cumulative_reward(df)

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
def dqn_rnd_cumulative_reward():
    data = get_data("dqn_agent")
    df = data["rnd"]["data"]
    
    plot_cumulative_reward(df)

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
def dyna_cumulative_reward():
    data = get_data("dyna_agent")
    df = data["dyna"]["data"]
    
    plot_cumulative_reward(df)

# PLOT #13 (4.4)
def dyna_cumulative_success():
    data = get_data("dyna_agent")
    df = data["dyna"]["data"]

    plot_cumulative_success(df)

# PLOT #14 (4.4)
def dyna_loss():
    data = get_data("dyna_agent")
    df = data["dyna"]["data"]

    raise NotImplementedError

# PLOT #15 (4.4)
def dyna_Q_values():
    # need to load Q table from file
    raise NotImplementedError

# PLOT #16 (4.4)
def dyna_key_episodes():
    raise NotImplementedError

# PLOT #17 (4.4) (bonus)
def dyna_key_Q_values():
    raise NotImplementedError

# PLOT #18 (4.5) 
def comparison_env_rewards():
    raise NotImplementedError

# PLOT #19 (4.5)
def comparison_eval_performance():
    raise NotImplementedError


    

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
    # dyna_loss()
    # dyna_Q_values()
    # dyna_key_episodes()
    # dyna_key_Q_values()
    # comparison_env_rewards()
    # comparison_eval_performance()

    plt.show()
    