from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data_from_file(data_path):
    # Data is stored as a dict with keys 'x' and 't' into a file with path='path_data'
    # The values of the keys are x_data and t_data:
    # x_data: list of length num_data_train, each element is numpy array of shape T_i (for categorical)
    # t_data: list of length num_data_train, each element is numpy array of shape T_i x 2
    with open(data_path, 'rb') as handle:
        data = pickle.load(handle)
    return data['x'], data['t']


def get_dataloader(data_path, marker_type, batch_size):
    if marker_type == "categorical":
        dataset = DSetCategorical(data_path)
        dataloader = DLoaderCategorical(dataset, bs=batch_size)
    else:
        raise NotImplementedError
    return dataloader


def collate_fn_categorical_marker(xt_tuples):
    """
    Input: list of (x_i,t_i) tuples
    Output: tensors (x, t, m) of shapes (max_len, BS), (max_len, BS, t_dim), (max_len, BS)
    Note: the x_tensor will contain the class labels, and not one-hot representation
    """
    BS = len(xt_tuples)

    x_data = [x for x, t in xt_tuples]
    t_data = [t for x, t in xt_tuples]

    t_dim = t_data[0].shape[1]

    seq_len = [len(t) for t in t_data]
    max_seq_len = max(seq_len)

    x_tensor = np.zeros((BS, max_seq_len))
    t_tensor = np.zeros((BS, max_seq_len, t_dim))
    mask_tensor = np.zeros((BS, max_seq_len))

    for idx in range(BS):
        x_tensor[idx, :seq_len[idx]] = x_data[idx].flatten()
        t_tensor[idx, :seq_len[idx]] = t_data[idx]
        mask_tensor[idx, :seq_len[idx]] = 1.

    x_tensor, t_tensor, mask_tensor = torch.tensor(x_tensor).long().to(device), torch.tensor(t_tensor).float().to(
        device), torch.tensor(mask_tensor).float().to(device)
    x_tensor, t_tensor, mask_tensor = [torch.transpose(_tensor, 0, 1) for _tensor in (x_tensor, t_tensor, mask_tensor)]
    return x_tensor.contiguous(), t_tensor.contiguous(), mask_tensor.contiguous()


def collate_fn_real_marker(xt_tuples):
    """
    Input: list of (x_i,t_i) tuples
    Output: tensors (x, t, m) of shapes (BS, max_len, x_dim), (BS, max_len, t_dim), (BS, max_len)
    """
    BS = len(xt_tuples)

    x_data = [x for x, t in xt_tuples]
    t_data = [t for x, t in xt_tuples]

    x_dim = x_data[0].shape[1]
    t_dim = t_data[0].shape[1]

    seq_len = [len(t) for t in t_data]
    max_seq_len = max(seq_len)

    x_tensor = np.zeros((BS, max_seq_len, x_dim))
    t_tensor = np.zeros((BS, max_seq_len, t_dim))
    mask_tensor = np.zeros((BS, max_seq_len))

    for idx in range(BS):
        x_tensor[idx, :seq_len[idx]] = x_data[idx]
        t_tensor[idx, :seq_len[idx]] = t_data[idx]
        mask_tensor[idx, :seq_len[idx]] = 1.

    x_tensor, t_tensor, mask_tensor = torch.tensor(x_tensor).long().to(device), torch.tensor(t_tensor).float().to(
        device), torch.tensor(mask_tensor).float().to(device)
    return (x_tensor, t_tensor, mask_tensor)


################################ DataSets ################################

class DSet(Dataset):
    def __init__(self, dataset_path):
        self.x_data, self.t_data = load_data_from_file(dataset_path)

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


class DSetCategorical(DSet):
    def __init__(self, dset_path):
        super().__init__(dset_path)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.t_data[idx]


class SingleSequenceDSetCategorical(DSetCategorical):
    def __init__(self, dataset_path, sequence_idx=0):
        super(SingleSequenceDSetCategorical, self).__init__(dataset_path)
        self.x_data = self.x_data[sequence_idx:sequence_idx + 1]
        self.t_data = self.t_data[sequence_idx:sequence_idx + 1]

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.t_data[idx]


class DLoaderCategorical(DataLoader):
    marker_type = 'categorical'

    def __init__(self, dset, bs=16):
        super().__init__(dset, batch_size=bs, collate_fn=collate_fn_categorical_marker)


if __name__ == "__main__":
    data_dir = Path("../../data")  # wrt utils folder
    # MIMIC
    # data_path = data_dir / 'mimic2_1_train.pkl'
    # marker_type = 'categorical'
    # loader = get_dataloader(data_path=data_path, marker_type=marker_type, batch_size=16)

    # Synthetic - Hawkes
    data_path = data_dir / 'simulated_hawkes_1_train.pkl'
    marker_type = 'categorical'
    dataset = SingleSequenceDSetCategorical(data_path)
    loader = DLoaderCategorical(dataset, bs=16)
    import pdb

    pdb.set_trace()

    for b_idx, (input_x, input_t, input_mask) in enumerate(loader):
        pass
