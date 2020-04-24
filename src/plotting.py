from abc import abstractmethod, ABC

import matplotlib
import numpy
import torch
from matplotlib.axes import Axes, np

from data_model_sandbox import HawkesProcessDataModelSandBox, RMTPPDataModelSandBox, Model1DataModelSandBox
from hyperparameters import RMTPPHyperParams, HawkesHyperparams, Model1HyperParams
from parameters import PlottingParams, DataModelParams, DataParams

matplotlib.use('agg')
from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BasePlot:
    def __init__(self, plotting_params: PlottingParams):
        self.plotting_params = plotting_params
        self._figure_dir = self.plotting_params.get_plotting_dir()

    def save_plot_and_close_fig(self, fig: plt.Figure, file_name: str):
        figure_path = self._figure_dir / (file_name + ".pdf")
        fig.savefig(figure_path)
        print(f"Saved plot to {figure_path}")
        plt.close(fig)


class BasePlotter:

    @abstractmethod
    def _get_intensity(self, sequence_idx, grid_size):
        """Computes the intensity of a sequence"""

    @staticmethod
    def _plot_data_timestamps(data_timestamps, ax: Axes):
        ax.scatter(data_timestamps, [0] * len(data_timestamps), s=1)


class HawkesPlotter(BasePlotter):
    def __init__(self, data_model_sandbox: HawkesProcessDataModelSandBox):
        self.data_model_sandbox = data_model_sandbox

    def _get_intensity(self, sequence_idx, grid_size):
        data_timestamps = self.data_model_sandbox.setup(sequence_idx)
        intensity, evaluated_timesteps = self.data_model_sandbox.get_intensity_over_grid(data_timestamps,
                                                                                         grid_size=grid_size)
        return intensity, evaluated_timesteps, data_timestamps

    def plot_intensity_vs_timestamp_to_axes(self, ax: Axes, sequence_idx=0, grid_size=1000):
        intensity_vals, t_vals, data_timestamps = self._get_intensity(sequence_idx, grid_size)
        ax.plot(t_vals, intensity_vals, label='hawkes')
        self._plot_data_timestamps(data_timestamps, ax)

    def plot_intensity_vs_time_index_to_axes(self, ax: Axes, sequence_idx=0):
        intensity_vals, _, data_timestamps = self._get_intensity(sequence_idx, None)
        ax.plot(np.arange(len(intensity_vals)), intensity_vals, label='hawkes')
        self._plot_data_timestamps(data_timestamps, ax)

    @classmethod
    def from_hyperparams(cls, hyperparams_dict, params, split_name):
        data_params = DataParams(params=params, split_name=split_name)
        model_hyperparams = HawkesHyperparams(**hyperparams_dict)
        data_model_sandbox = HawkesProcessDataModelSandBox(data_params, model_hyperparams)
        return cls(data_model_sandbox)


class BaseNNPlotter(BasePlotter):
    data_model_sandbox_class = None
    hyperparams_class = None
    model_name = None

    def __init__(self, data_model_sandbox):
        self.data_model_sandbox = data_model_sandbox

    def plot_intensity_vs_timestamp_to_axes(self, ax: Axes, sequence_idx=0, grid_size=1000):
        intensity_vals, t_vals, data_timestamps = self._get_intensity(sequence_idx, grid_size)
        ax.plot(t_vals, intensity_vals, label=self.model_name)
        self._plot_data_timestamps(data_timestamps, ax)

    def plot_intensity_vs_time_index_to_axes(self, ax: Axes, sequence_idx=0):
        intensity_vals, _, data_timestamps = self._get_intensity(sequence_idx, None)
        ax.plot(np.arange(len(intensity_vals)), intensity_vals, label=self.model_name)
        self._plot_data_timestamps(data_timestamps, ax)

    def _get_intensity(self, sequence_idx: int, grid_size: int):
        log_intensity, data_timestamps, evaluated_timestamps = self.data_model_sandbox.get_intensity_over_grid(
            sequence_idx, grid_size)
        intensity_np = log_intensity.exp().detach().cpu().numpy().flatten()
        evaluated_timestamps_np = evaluated_timestamps.cpu().numpy().flatten()
        data_timestamps_np = data_timestamps.cpu().numpy().flatten()
        return intensity_np, evaluated_timestamps_np, data_timestamps_np

    @classmethod
    def from_filename(cls, model_filename, params, split_name):
        data_model_params = DataModelParams(
            params=params,
            model_filename=model_filename,
            split_name=split_name
        )
        model_hyperparams = cls.hyperparams_class(params)
        data_model_sandbox = cls.data_model_sandbox_class(data_model_params, model_hyperparams)
        return cls(data_model_sandbox)


class RMTPPPlotter(BaseNNPlotter):
    data_model_sandbox_class = RMTPPDataModelSandBox
    hyperparams_class = RMTPPHyperParams
    model_name = "RMTPP"


class Model1Plotter(BaseNNPlotter):
    data_model_sandbox_class = Model1DataModelSandBox
    hyperparams_class = Model1HyperParams
    model_name = "Model1"


class MultipleModelPlot(BasePlot):
    def __init__(self, plotting_params, plotter_list):
        super(MultipleModelPlot, self).__init__(plotting_params)
        self.plotter_list = plotter_list
        self.fig, self.ax = plt.subplots(1, 1)

    def plot(self, plot_title, sequence_idx):
        for plotter in self.plotter_list:
            self._make_single_plot(plotter, sequence_idx)
        self.ax.set_title(plot_title)
        self.ax.legend()

    def _make_single_plot(self, plotter, sequence_idx):
        assert isinstance(plotter, (BaseNNPlotter, HawkesPlotter))
        plotter.plot_intensity_vs_time_index_to_axes(self.ax, sequence_idx)

    @classmethod
    def from_params(cls, params, plotter_list):
        plotting_params = PlottingParams(params)
        return cls(plotting_params, plotter_list)
