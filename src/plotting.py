import os
import argparse
from argparse import Namespace

import numpy as np
import torch

import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

from utils.data_loader import get_dataloader
from utils.model_loader import ModelLoader
from main import setup_parser, _augment_params

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Plotter():
    def __init__(self):
        self.model = None
        self.model_path = None
        self.dataloader = None

    def load_data(self, params: Namespace, split_name):
        split_name = "train"  # TODO remove after testing
        # Data should reside in this path for all datasets. Ideally 5 cross fold validation.
        data_path = params.data_dir + params.data_name + '_' + str(params.cv_idx) + f"_{split_name}.pkl"
        self.dataloader = get_dataloader(data_path, params.marker_type, params.batch_size)

    def load_model(self, params: Namespace, model_state_path: str):
        loader = ModelLoader(params, model_state_path=model_state_path)
        self.model = loader.model
        self.model_path = loader.model_state_path


class HawkesPlotter(Plotter):
    def __init__(self):
        super().__init__()
        self.params = {'lambda_0': .2, 'alpha': .8, 'beta': 1.}

    def _get_intensity(self, t: float, hist: np.ndarray, params: dict):
        # params: [lambda0, alpha, beta]
        # hist must be numpy array
        hist = hist[(hist < t)]
        return params['lambda_0'] + params['alpha'] * np.sum(np.exp(-1. * (t - hist) / params['beta']))

    def plot_intensity(self, ax: plt.Axes):
        assert self.dataloader is not None
        data = self.dataloader.dataset[0]
        x_vals = np.arange(0, data[-1], .5)
        y_vals = [self._get_intensity(t=_t, hist=data, params=self.params) for _t in x_vals]
        ax.plot(x_vals, y_vals)


if __name__ == "__main__":
    parser = setup_parser()
    params = parser.parse_args()
    ###
    params.model = 'rmtpp'
    params.data_name = 'simulated_hawkes'
    ###
    _augment_params(params)

    plotter = HawkesPlotter()

    # Create data loader
    plotter.load_data(params, "train")

    # Load model object & Load model state
    model_state_path = os.path.join('model', params.data_name, params.model,
                                    "_g1_do0.5_b32_h256_l20.0_l20_gn10.0_lr0.001_c10_s1_tlintensity_ai40")
    plotter.load_model(params, model_state_path)

    # Plot
    fig, ax = plt.subplots(1, 1)
    plotter.plot_intensity()
