import numpy as np
from pathlib import Path
import pickle

from helper import train_val_split


def read_timestamps(filename_times):
    # could have just used np.loadtxt but made it more general
    # because that might not have worked with uneven-length-sequences
    data = []
    with open(filename_times, 'r') as fobj:
        for line in fobj:
            line = line.rstrip().split(' ')
            data.append([float(n) for n in line])
    return data


def read_events(filename_events):
    # could have just used np.loadtxt but made it more general
    # because that might not have worked with uneven-length-sequences
    data = []
    with open(filename_events, 'r') as fobj:
        for line in fobj:
            line = line.rstrip().split(' ')
            data.append([int(n) for n in line])
    return data


def _write_file(data_obj, data_path):
    with open(data_path, 'wb') as fobj:
        pickle.dump(data_obj, fobj)
    print(f"File written to {data_path}")


def write_data(data: dict, dir_path: Path, dataset_name: str, split_name: str):
    out_path = dir_path / (dataset_name + "_" + split_name + ".pkl")
    _write_file(data, out_path)


def _process_split(dir_path, split_name):
    times_path = dir_path / f"time-{split_name}.txt"
    events_path = dir_path / f"event-{split_name}.txt"

    data_timestamps = read_timestamps(times_path)
    data_events = read_events(events_path)

    data_timestamps = [[d_i_t - min(d_i) for d_i_t in d_i] for d_i in data_timestamps]  # first time is 0 by convention
    intervals = [np.diff([0] + d_i) for d_i in data_timestamps]  # first interval is 0 by convention
    data_times = [np.stack([int_i, ts_i], axis=-1) for int_i, ts_i in zip(intervals, data_timestamps)]

    # Required output format:
    # type: dict
    # keys: 't', 'x'
    # x_data: list of length num_data_train, each element is numpy array of shape T_i (for categorical)
    # t_data: list of length num_data_train, each element is numpy array of shape T_i x 2 (intervals, timestamps)
    data_processed = dict()
    data_processed['t'] = data_times
    data_processed['x'] = [np.array(d_i) for d_i in data_events]
    return data_processed


def get_dataset_info():
    info_dict = {
        "hawkes": {
            "dataset_name": "simulated_hawkes_1",
            "dir": "/home/abhishekshar/hierarchichal_point_process/data/rmtpp_synthetic/hawkes"
        }
    }
    return info_dict


def process_data():
    split_names = ["train", "test"]
    dataset_info_dict = get_dataset_info()

    out_dir = Path("/home/abhishekshar/hierarchichal_point_process/data")

    for _, dataset_info in dataset_info_dict.items():
        dataset_name = dataset_info["dataset_name"]
        dir_path = Path(dataset_info["dir"])

        for split_name in split_names:
            data_processed = _process_split(dir_path, split_name)

            if split_name == "train":
                data_train, data_val = train_val_split(data_processed, val_ratio=0.2)
                write_data(data_train, out_dir, dataset_name, "train")
                write_data(data_val, out_dir, dataset_name, "valid")
            else:
                write_data(data_train, out_dir, dataset_name, split_name)


if __name__ == "__main__":
    process_data()
