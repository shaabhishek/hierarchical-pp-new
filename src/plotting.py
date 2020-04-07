from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

import matplotlib

from data_model_sandbox import RMTPPDataModelSandBox, HawkesProcessDataModelSandBox
from parameters import DataModelParams, RMTPPHyperparams, PlottingParams, HawkesHyperparams, DataParams
from utils.helper import make_intermediate_dirs_if_absent

matplotlib.use('agg')
import matplotlib.pyplot as plt

from utils.data_loader import get_dataloader
from utils.model_loader import ModelLoader
from main import setup_parser, _augment_params

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BasePlotter():
    def __init__(self, plotting_params: PlottingParams):
        self.plotting_params = plotting_params
        self._figure_dir = self.plotting_params.get_plotting_dir()
        make_intermediate_dirs_if_absent(self._figure_dir)

    def save_plot_and_close_fig(self, fig: plt.Figure, file_name: str):
        figure_path = self._figure_dir / (file_name + ".png")
        fig.savefig(figure_path)
        print(f"Saved plot to {figure_path}")
        plt.close(fig)


class HawkesPlotter(BasePlotter):
    def __init__(self, data_model_sandbox: HawkesProcessDataModelSandBox, plotting_params: PlottingParams):
        super().__init__(plotting_params)
        self.data_model_sandbox = data_model_sandbox

    def _get_intensity(self):
        data_timestamps = self.data_model_sandbox.setup(0)
        intensity, evaluated_timesteps = self.data_model_sandbox.get_intensity_over_grid(data_timestamps)
        return intensity, evaluated_timesteps, data_timestamps

    def plot_intensity_to_axes(self, ax: plt.Axes):
        intensity_vals, t_vals, data_timestamps = self._get_intensity()
        ax.plot(t_vals, intensity_vals)
        ax.scatter(data_timestamps, [min(intensity_vals)] * len(data_timestamps), s=1)
        # if t_range is not None:
        #     ax.set_xlim(t_range)
        ax.set_title('Hawkes True Model Intensity')


class RMTPPPlotter(BasePlotter):
    def __init__(self, data_model_sandbox, plotting_params: PlottingParams):
        super().__init__(plotting_params)
        self.data_model_sandbox = data_model_sandbox

    def _get_intensity(self):
        """"""
        hidden_seq, data_timestamps = self.data_model_sandbox.setup(0)  # # (T, BS=1, h_dim), (T, BS=1, 1)

        log_intensity, evaluated_timestamps = self.data_model_sandbox.get_intensity_over_grid(hidden_seq,
                                                                                              data_timestamps)
        intensity_np = log_intensity.exp().detach().cpu().numpy().flatten()
        evaluated_timestamps_np = evaluated_timestamps.cpu().numpy().flatten()
        data_timestamps_np = data_timestamps.cpu().numpy().flatten()
        return intensity_np, evaluated_timestamps_np, data_timestamps_np

    def plot_intensity_to_axes(self, ax: plt.Axes):
        intensity_vals, t_vals, data_timestamps = self._get_intensity()
        ax.plot(t_vals, intensity_vals)
        ax.scatter(data_timestamps, [min(intensity_vals)] * len(data_timestamps), s=1)
        ax.set_title('RMTPP Intensity')


def test_rmtpp(params: Namespace):
    data_model_params = DataModelParams(
        params=params,
        model_file_identifier='_g1_do0.5_b32_h256_l20.0_l20_gn10.0_lr0.001_c10_s1_tlintensity_ai40',
        split_name='train'
    )
    model_hyperparams = RMTPPHyperparams(params)
    plotting_params = PlottingParams(params)

    data_model_sandbox = RMTPPDataModelSandBox(data_model_params, model_hyperparams)
    plotter = RMTPPPlotter(data_model_sandbox, plotting_params)

    fig, ax = plt.subplots(1, 1)
    plotter.plot_intensity_to_axes(ax)
    plotter.save_plot_and_close_fig(fig, "test_hawkes_rmtpp_idx0")


def test_hawkes_truemodel(params: Namespace):
    data_params = DataParams(params=params, split_name='train')
    model_hyperparams = HawkesHyperparams(**{'lambda_0': .2, 'alpha': .8, 'beta': 1.})
    plotting_params = PlottingParams(params)

    data_model_sandbox = HawkesProcessDataModelSandBox(data_params, model_hyperparams)
    hawkes_plotter = HawkesPlotter(data_model_sandbox, plotting_params)

    fig, ax = plt.subplots(1, 1)
    hawkes_plotter.plot_intensity_to_axes(ax)
    hawkes_plotter.save_plot_and_close_fig(fig, "test_hawkes_truemodel_idx0")


if __name__ == "__main__":
    parser = setup_parser()
    params = parser.parse_args()
    ###
    params.model = 'rmtpp'
    params.data_name = 'simulated_hawkes'
    ###
    _augment_params(params)

    test_rmtpp(params)
    test_hawkes_truemodel(params)

    # Create data loader
    # plotter.load_data(params, "train")

    # Load model object & Load model state
    # model_state_path = os.path.join('model', params.data_name, params.model,
    #                                 "_g1_do0.5_b32_h256_l20.0_l20_gn10.0_lr0.001_c10_s1_tlintensity_ai40")
    # plotter.load_model(params, model_state_path)
