from abc import abstractmethod, ABC

import matplotlib
import torch
from matplotlib.axes import Axes, np

from data_model_sandbox import HawkesProcessDataModelSandBox, RMTPPDataModelSandBox
from hyperparameters import RMTPPHyperparams, HawkesHyperparams
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


class BasePlotter(ABC):

    @abstractmethod
    def _get_intensity(self, sequence_idx, grid_size):
        """Computes the intensity of a sequence"""

    @staticmethod
    def _plot_data_timestamps(data_timestamps, ax: Axes):
        ax.scatter(data_timestamps, [0] * len(data_timestamps), s=1)

    @staticmethod
    def _make_plot(ax, plot_title):
        ax.set_title(plot_title)


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
        self._make_plot(ax, plot_title='Hawkes True Model: Intensity vs Time')

    def plot_intensity_vs_time_index_to_axes(self, ax: Axes, sequence_idx=0):
        intensity_vals, _, data_timestamps = self._get_intensity(sequence_idx, None)
        ax.plot(np.arange(len(intensity_vals)), intensity_vals, label='hawkes')
        self._plot_data_timestamps(data_timestamps, ax)
        self._make_plot(ax, plot_title='Hawkes True Model: Intensity vs Time')

    @classmethod
    def from_hyperparams(cls, hyperparams_dict, params, split_name):
        data_params = DataParams(params=params, split_name=split_name)
        model_hyperparams = HawkesHyperparams(**hyperparams_dict)
        data_model_sandbox = HawkesProcessDataModelSandBox(data_params, model_hyperparams)
        return cls(data_model_sandbox)


class RMTPPPlotter(BasePlotter):
    def __init__(self, data_model_sandbox: RMTPPDataModelSandBox):
        self.data_model_sandbox = data_model_sandbox

    def _get_intensity(self, sequence_idx, grid_size):
        """
        :param grid_size:
        """
        hidden_seq, data_timestamps = self.data_model_sandbox.setup(sequence_idx)  # # (T, BS=1, h_dim), (T, BS=1, 1)

        log_intensity, evaluated_timestamps = self.data_model_sandbox.get_intensity_over_grid(hidden_seq,
                                                                                              data_timestamps,
                                                                                              grid_size=grid_size)
        intensity_np = log_intensity.exp().detach().cpu().numpy().flatten()
        evaluated_timestamps_np = evaluated_timestamps.cpu().numpy().flatten()
        data_timestamps_np = data_timestamps.cpu().numpy().flatten()
        return intensity_np, evaluated_timestamps_np, data_timestamps_np

    def plot_intensity_vs_timestamp_to_axes(self, ax: Axes, sequence_idx=0, grid_size=1000):
        intensity_vals, t_vals, data_timestamps = self._get_intensity(sequence_idx, grid_size)

        ax.plot(t_vals, intensity_vals, label='rmtpp')
        self._plot_data_timestamps(data_timestamps, ax)

        self._make_plot(ax, 'RMTPP: Intensity vs Time')

    def plot_intensity_vs_time_index_to_axes(self, ax: Axes, sequence_idx=0):
        intensity_vals, _, data_timestamps = self._get_intensity(sequence_idx, None)

        ax.plot(np.arange(len(intensity_vals)), intensity_vals, label='rmtpp')
        self._plot_data_timestamps(data_timestamps, ax)

        self._make_plot(ax, 'RMTPP: Intensity vs Time Index')

    @classmethod
    def from_filename(cls, model_filename, params, split_name):
        data_model_params = DataModelParams(
            params=params,
            model_filename=model_filename,
            split_name=split_name
        )
        model_hyperparams = RMTPPHyperparams(params)
        data_model_sandbox = RMTPPDataModelSandBox(data_model_params, model_hyperparams)
        return cls(data_model_sandbox)


class IntensityVsTimeIndexPlotMixin:
    def _make_single_plot(self, plotter):
        assert isinstance(self, MultipleModelPlot)
        assert isinstance(plotter, (RMTPPPlotter, HawkesPlotter))
        plotter.plot_intensity_vs_time_index_to_axes(self.ax, sequence_idx=10)


class MultipleModelPlot(BasePlot):
    def __init__(self, plotting_params, plotter_list):
        super(MultipleModelPlot, self).__init__(plotting_params)
        self.plotter_list = plotter_list
        self.fig, self.ax = plt.subplots(1, 1)

    def plot(self, plot_title):
        for plotter in self.plotter_list:
            self._make_single_plot(plotter)
        self.ax.set_title(plot_title)
        self.ax.legend()

    @classmethod
    def from_params(cls, params, plotter_list):
        plotting_params = PlottingParams(params)
        return cls(plotting_params, plotter_list)


class RMTPPHawkesIntensityTimeIndexPlot(IntensityVsTimeIndexPlotMixin, MultipleModelPlot):
    pass