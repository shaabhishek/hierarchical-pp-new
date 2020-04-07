from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

import matplotlib

from data_model_sandbox import DataModelSandBox
from parameters import ModelHyperparams, DataModelParams

matplotlib.use('agg')
import matplotlib.pyplot as plt

from utils.data_loader import get_dataloader
from utils.model_loader import ModelLoader
from main import setup_parser, _augment_params

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BasePlotter():
    def __init__(self):
        self.timestamps = None

    # def load_data(self, params: Namespace, split_name):
    #     split_name = "train"  # TODO remove after testing
    #     # Data should reside in this path for all datasets. Ideally 5 cross fold validation.
    #     data_path = params.data_dir + params.data_name + '_' + str(params.cv_idx) + f"_{split_name}.pkl"
    #     self.dataloader = get_dataloader(data_path, params.marker_type, params.batch_size)
    #     self.timestamps = [t_sequence[:, 1] for t_sequence in self.dataloader.dataset.t_data]


class Plotter():
    def __init__(self, plot_dir: Path):
        self.model = None
        self.model_path = None
        self.dataloader = None
        self.plot_dir = plot_dir

    def load_data(self, params: Namespace, split_name):
        split_name = "train"  # TODO remove after testing
        # Data should reside in this path for all datasets. Ideally 5 cross fold validation.
        data_path = params.data_dir + params.data_name + '_' + str(params.cv_idx) + f"_{split_name}.pkl"
        self.dataloader = get_dataloader(data_path, params.marker_type, params.batch_size)

    def load_model(self, params: Namespace, model_state_path: str):
        loader = ModelLoader(params, model_state_path=model_state_path)
        self.model = loader.model
        self.model_path = loader.model_state_path

    def save_plot(self, fig: plt.Figure):
        plt.savefig()


class HawkesPlotter(BasePlotter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params = {'lambda_0': .2, 'alpha': .8, 'beta': 1.}

    def _get_intensity(self, t: float, hist: np.ndarray, params: dict):
        # params: [lambda0, alpha, beta]
        # hist must be numpy array
        hist = hist[(hist < t)]
        return params['lambda_0'] + params['alpha'] * np.sum(np.exp(-1. * (t - hist) / params['beta']))

    def plot_intensity(self, ax: plt.Axes, t_range: list = None):
        assert self.timestamps is not None
        data = self.timestamps[0]
        t_vals = np.arange(0, data[-1], .5)
        intensity_vals = [self._get_intensity(t=_t, hist=data, params=self.params) for _t in t_vals]

        ax.plot(t_vals, intensity_vals)
        ax.scatter(data, [min(intensity_vals)] * len(data), s=1)
        if t_range is not None:
            ax.set_xlim(t_range)
        ax.set_title('Data till t=100')


class RMTPPPlotter(BasePlotter):
    def __init__(self, data_model_sandbox):
        super().__init__()
        self.data_model_sandbox = data_model_sandbox

    def _get_intensity(self):
        """"""
        hidden_seq, data_timestamps = self.data_model_sandbox.setup(0)

        log_intensity, evaluated_timestamps = self.data_model_sandbox.get_intensity_over_grid(hidden_seq, data_timestamps)
        intensity_np = log_intensity.exp().detach().cpu().numpy().flatten()
        timestamps_np = evaluated_timestamps.cpu().numpy().flatten()
        return intensity_np, timestamps_np

    def plot_intensity(self, ax: plt.Axes, t_range: list = None):
        intensity_vals, t_vals = self._get_intensity()
        import pdb; pdb.set_trace()


def test_rmtpp(params: Namespace):
    data_model_params = DataModelParams(
        params, model_file_identifier="_g1_do0.5_b32_h256_l20.0_l20_gn10.0_lr0.001_c10_s1_tlintensity_ai40")
    model_hyperparams = ModelHyperparams(params)
    data_model_sandbox = DataModelSandBox(data_model_params, model_hyperparams, 'train')
    plotter = RMTPPPlotter(data_model_sandbox)

    fig, ax = plt.subplots(1,1)
    plotter.plot_intensity(ax)

if __name__ == "__main__":
    parser = setup_parser()
    params = parser.parse_args()
    ###
    params.model = 'rmtpp'
    params.data_name = 'simulated_hawkes'
    ###
    _augment_params(params)

    test_RMTPP(params)

    # plotting_params = PlottingParams(params)
    # plotter = HawkesPlotter(plot_dir=plotting_params.get_plotting_dir())

    # Create data loader
    # plotter.load_data(params, "train")

    # Load model object & Load model state
    # model_state_path = os.path.join('model', params.data_name, params.model,
    #                                 "_g1_do0.5_b32_h256_l20.0_l20_gn10.0_lr0.001_c10_s1_tlintensity_ai40")
    # plotter.load_model(params, model_state_path)



    # Plot
    fig, ax = plt.subplots(1, 1)
    plotter.plot_intensity(ax)
