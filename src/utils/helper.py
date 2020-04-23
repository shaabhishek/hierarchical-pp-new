# from progress.bar import Bar
import os
from pathlib import Path

import torch
from progressbar import bar
import numpy as np


class ProgressBar(bar.ProgressBar):
    def __init__(self, label, max):
        super().__init__(max_value=max, prefix=label)
        # Makes sure that the next() function works on the function
        self._iterable = iter(range(max))


def train_val_split(data, val_ratio=0.2):
    """
        Input:
            data: dict with keys 'x' and 't'.
            data['x']: x_data: list of length num_data_train, each element is numpy array of shape
            - [T_i x marker_dim] (if real)
            - [T_i,] (if categorical)
            data['t']: t_data: list of length num_data_train, each element is numpy array of shape T_i x 2
        Output:
            train_split, val_split: both follow the same structure as data
    """
    N = len(data['x'])
    val_size = int(val_ratio * N)

    random_order = np.arange(N)
    np.random.shuffle(random_order)

    train_split = {}
    val_split = {}
    for key, value in data.items():
        train_split[key] = [data[key][i] for i in random_order[:N - val_size]]
        val_split[key] = [data[key][i] for i in random_order[N - val_size:]]

    return train_split, val_split


def make_intermediate_dirs_if_absent(full_file_path: Path):
    """Make intermediate folders, and don't throw error if they exist"""
    os.makedirs(str(full_file_path.resolve()), exist_ok=True)


def _prepend_dims_to_tensor(tensor, *dims):
    """Add dimensions of required size in the beginning of tensor
    Example:
    _prepend_dims(tensor, d1, d2) modifies tensor of shape (a,b,c) to tensor of shape (d1, d2, a, b, c)
    """
    return tensor.view(*[1 for _ in dims], *tensor.shape).expand(*dims, *tensor.shape)


def assert_shape(variable_name: str, variable_shape: torch.Size, expected_shape: tuple):
    assert variable_shape == expected_shape, f"Shape of {variable_name} is incorrect: {variable_shape}, Expected: {expected_shape} "


def pretty_print_table(*args):
    for arg in args:
        print(f"{str(arg):>60}", end='')
    print()
