from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
from tensorboard.compat.proto import event_pb2
import time

import traceback
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

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


# Extraction function
def tflog2pandas(path):
    """
    Code taken from https://stackoverflow.com/a/71240906/20694757
    """
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, size_guidance={"scalars": 0})
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data

