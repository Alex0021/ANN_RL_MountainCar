import os, sys
from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
from tensorboard.compat.proto import event_pb2
import matplotlib.pyplot as plt
import numpy as np
import time


def extract_data_from_event(e):
    data_name = e.summary.value[0].tag.split('/')[-1]
    value = e.summary.value[0].simple_value
    return data_name, int(e.step), float(value)

def extract_data_from_tb_file(path):
    loader = RawEventFileLoader(path)
    data_dict = {}
    info = "Reading data"
    start_time = time.time()
    for raw_event in loader.Load():
        print(info, end="\r")
        e = event_pb2.Event.FromString(raw_event)
        try:
            data_name, step, value = extract_data_from_event(e)
            if data_name not in data_dict:
                data_dict[data_name] = []
            data_dict[data_name].append((step, value))
        except:
            pass    
        if time.time() - start_time > 1:
            info += "."
            start_time = time.time()
    print()
    return data_dict

def plot_loss(data_dict):
    if 'loss' not in data_dict:
        print("No loss data found in dictionary")
        return
    print("Plotting loss")
    data = np.array(data_dict['loss'])
    update_steps = [d[0] for d in data]
    loss_values = [d[1] for d in data]
    avg = 0
    loss_values_avg = np.zeros_like(loss_values)
    for i in range(len(loss_values)):
        avg = avg + (loss_values[i] - avg) / (i+1)
        loss_values_avg[i] = avg
    plt.figure()
    plt.plot(update_steps, loss_values)
    plt.title("Loss (MSE) per update step")
    plt.xlabel("Update steps")
    plt.ylabel("Loss")
    # Add average
    plt.plot(update_steps, loss_values_avg, label="Average Loss", color='orange')
    plt.legend()

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

def plot_cumulative_success(data_dict):
    if 'episode_completed' not in data_dict:
        print("No episode completed data found in dictionary")
        return
    print("Plotting episode completed")
    data = np.array(data_dict['episode_completed'])
    episode_steps = [d[0] for d in data]
    cumulative_success = np.cumsum([d[1] for d in data])
    plt.figure()
    plt.plot(episode_steps, cumulative_success, linewidth=1.5)
    plt.title("Cumulative successes over episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative successes")

if __name__ == "__main__":
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

            