import torch
from matplotlib.axes import Axes, np
from torch import nn
from torch.distributions import Categorical, Normal, Gamma
from torch.utils.data import Dataset

from base_model import DistSampleTuple


class Poly(nn.Module):
    def __init__(self, c):
        super(Poly, self).__init__()
        self.c = c

    def forward(self, x):
        return x ** self.c


class AddConstant(nn.Module):
    def __init__(self, c):
        super(AddConstant, self).__init__()
        self.c = c

    def forward(self, x):
        return x + self.c


class Constant(nn.Module):
    def __init__(self, c):
        super(Constant, self).__init__()
        self.c = c

    def forward(self, x):
        return torch.zeros_like(x) + self.c


class Scale(nn.Module):
    def __init__(self, c):
        super(Scale, self).__init__()
        self.c = c

    def forward(self, x):
        return x * self.c


class NegExpKernel(nn.Module):

    def forward(self, x):
        return torch.exp(-x)


class BaseRandomVariable:
    distribution = None

    def sample(self, *shape):
        return DistSampleTuple(self.distribution, self.distribution.sample(shape))

    def _get_parameters(self, *args):
        raise NotImplementedError


class YModel(BaseRandomVariable):
    def __init__(self):
        pi = self._get_parameters()
        self.distribution = Categorical(logits=pi)

    def _get_parameters(self):
        pi = torch.tensor([0.4, 0.6])
        return pi


class ZModel(BaseRandomVariable):
    def __init__(self, dim):
        self.dim = dim

    def _get_distribution(self, z_prev):
        if z_prev is None:
            mu, sigma = torch.zeros(self.dim), 20 * torch.ones(self.dim)
        else:
            mu_model, sigma_model = self._get_parameters()
            mu = mu_model(z_prev)
            sigma = sigma_model(z_prev)
        distribution = Normal(loc=mu, scale=sigma)
        return distribution

    def _get_parameters(self):
        mu_model = nn.Linear(self.dim, self.dim)
        nn.init.constant_(mu_model.weight, 1 / 3)
        nn.init.constant_(mu_model.bias, 1)
        sigma_model = Constant(1)
        return mu_model, sigma_model

    def sample(self, *shape, z_prev):
        distribution = self._get_distribution(z_prev)
        return DistSampleTuple(distribution, distribution.sample(shape))


class XModel(BaseRandomVariable):
    def __init__(self, dim, z_dim):
        self.dim = dim
        self.z_dim = z_dim

    def _get_distribution(self, x_prev, z, y):
        if x_prev is None:
            x_prev = torch.zeros(*z.shape[:-1], self.dim)
        x_model, z_model, y_model, rate_model = self._get_parameters()
        # print(x_model(x_prev), x_prev)
        # print(z_model(z), '\n', z, '\n\n')
        concentration = y_model(y)
        rate = rate_model(x_model(x_prev) + z_model(z))
        if torch.isinf(rate).any():
            import pdb;
            pdb.set_trace()
        return Gamma(concentration, rate=rate)

    def _get_parameters(self):
        # x_model = nn.Linear(self.dim, self.dim, bias=False)
        x_model = nn.Sequential(NegExpKernel())
        z_model = nn.Linear(self.z_dim, self.dim, bias=False)
        y_model = nn.Linear(1, self.dim)
        rate_model = nn.Sequential(nn.Sigmoid(), Scale(20), AddConstant(5))

        with torch.no_grad():
            z_model.weight.data = torch.tensor([[.1, .1, .1]])

        c = 250.
        nn.init.constant_(y_model.weight, c)
        nn.init.constant_(y_model.bias, c)
        return x_model, z_model, y_model, rate_model

    def sample(self, *shape, x_prev, z, y):
        distribution = self._get_distribution(x_prev, z, y)
        sample = distribution.sample(shape)
        if torch.isinf(sample).any():
            import pdb;
            pdb.set_trace()
        return DistSampleTuple(distribution, sample)


class ZSequenceModel:
    def __init__(self, dim, length):
        self.dim = dim
        self.sequence = self._build_sequence(length)

    def _build_sequence(self, length):
        return [ZModel(self.dim) for _ in range(length)]

    def sample(self, n):
        z_dist_sample = self.sequence[0].sample(n, z_prev=None)
        sequence_distributions = [z_dist_sample.dist]
        sequence_samples = [z_dist_sample.sample]
        for z in self.sequence[1:]:
            z_dist_sample = z.sample(z_prev=z_dist_sample.sample)
            sequence_distributions.append(z_dist_sample.dist)
            sequence_samples.append(z_dist_sample.sample)
        return DistSampleTuple(sequence_distributions, torch.stack(sequence_samples, dim=0))


class XSequenceModel:
    def __init__(self, dim, z_dim, length):
        self.dim = dim
        self.z_dim = z_dim
        self.sequence = self._build_sequence(length)

    def _build_sequence(self, length):
        return [XModel(self.dim, self.z_dim) for _ in range(length)]

    def sample(self, z_seq, y):
        x_dist_sample = self.sequence[0].sample(x_prev=None, z=z_seq[0], y=y)
        sequence_distributions = [x_dist_sample.dist]
        sequence_samples = [x_dist_sample.sample]
        for x, z in zip(self.sequence[1:], z_seq[1:]):
            x_dist_sample = x.sample(x_prev=x_dist_sample.sample, z=z, y=y)
            sequence_distributions.append(x_dist_sample.dist)
            sequence_samples.append(x_dist_sample.sample)
        return DistSampleTuple(sequence_distributions, torch.stack(sequence_samples, dim=0))

class Simulator:
    def __init__(self):
        self.z_dim = 3
        self.x_dim = 1

    def simulate(self, length, n):
        y = YModel()
        z_seq = ZSequenceModel(self.z_dim, length)
        x_seq = XSequenceModel(self.x_dim, self.z_dim, length)

        self.y_dist_samples = y.sample(n)
        self.z_dist_samples = z_seq.sample(n)
        self.x_dist_samples = x_seq.sample(self.z_dist_samples.sample, self.y_dist_samples.sample.unsqueeze(-1).float())

    def _get_simulated_data(self):
        return self.y_dist_samples, self.z_dist_samples, self.x_dist_samples

    def save_simulated_data(self):
        from pathlib import Path
        from utils.rmtpp_simulated_data_processing import write_data
        data_dict = dict(y_dist_samples=self.y_dist_samples, z_dist_samples=self.z_dist_samples,
                         x_dist_samples=self.x_dist_samples)
        data_dir = Path("../data")  # wrt src folder
        data_path = data_dir / 'simulated_data_1_train.pkl'
        write_data(data_dict, data_path)
        self._process_data()

    def _process_data(self, intervals, ):
        """Required output format:
            type: dict
            keys: 't', 'x'
            x_data: list of length num_data_train, each element is numpy array of shape T_i (for categorical)
            t_data: list of length num_data_train, each element is numpy array of shape T_i x 2 (intervals, timestamps)
            """
        data_processed = dict()
        data_processed['t'] = data_times
        data_processed['x'] = [np.array(d_i) for d_i in data_events]


class SimulatedDataset(Dataset):
    def __init__(self):
        y_dist_samples, z_dist_samples, x_dist_samples =

    def __len__(self):
        return


def plot_y_params(y_dist: Categorical, axes: Axes):
    data = y_dist.probs.unsqueeze(0)
    axes.imshow(data)


def plot_rates(x_dist_seq, dimension: int, axes: Axes, *sequence_idx):
    length_seq = len(x_dist_seq)
    rates_seq = [dist.rate for dist in x_dist_seq]
    for idx in sequence_idx:
        data = [rates_vector[idx, dimension] for rates_vector in rates_seq]
        axes.plot(np.arange(length_seq) + 1, data, label=f'Seq {idx}')


def plot_z(z_seq: torch.Tensor, dimension: int, axes: Axes, *sequence_idx):
    length_seq, _, _ = z_seq.shape
    for idx in sequence_idx:
        data = z_seq[:, idx, dimension]
        axes.plot(np.arange(length_seq) + 1, data, label=f'Seq {idx}')


def plot_x_timestamps(x_seq: torch.Tensor, dimension: int, axes: Axes, *sequence_idx):
    length_seq, _, _ = x_seq.shape
    for idx in sequence_idx:
        intervals = x_seq[:, idx, dimension]
        timestamps = torch.cumsum(intervals, 0)
        axes.plot(np.arange(length_seq) + 1, timestamps, label=f'Seq {idx}')


def plot_y(y, axes: Axes, *sequence_idx):
    for idx in sequence_idx:
        data = y[idx]
        axes.scatter(idx, data, s=10, label=f'Seq {idx}')
