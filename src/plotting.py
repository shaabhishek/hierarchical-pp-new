import matplotlib
import torch
from matplotlib.axes import Axes

from data_model_sandbox import HawkesProcessDataModelSandBox, RMTPPDataModelSandBox
from parameters import PlottingParams
from utils.helper import make_intermediate_dirs_if_absent

matplotlib.use('agg')
from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SavePlotMixin:
    def save_plot_and_close_fig(self, fig: plt.Figure, file_name: str):
        figure_path = self._figure_dir / (file_name + ".png")
        fig.savefig(figure_path)
        print(f"Saved plot to {figure_path}")
        plt.close(fig)


class BasePlotter(SavePlotMixin):
    def __init__(self, plotting_params: PlottingParams):
        self.plotting_params = plotting_params
        self._figure_dir = self.plotting_params.get_plotting_dir()

    @staticmethod
    def _plot_data_timestamps(data_timestamps, ax: Axes):
        ax.scatter(data_timestamps, [0] * len(data_timestamps), s=1)


class HawkesPlotter(BasePlotter):
    def __init__(self, data_model_sandbox: HawkesProcessDataModelSandBox, plotting_params: PlottingParams):
        super().__init__(plotting_params)
        self.data_model_sandbox = data_model_sandbox

    def _get_intensity(self, sequence_idx):
        data_timestamps = self.data_model_sandbox.setup(0)
        intensity, evaluated_timesteps = self.data_model_sandbox.get_intensity_over_grid(data_timestamps)
        return intensity, evaluated_timesteps, data_timestamps

    def plot_intensity_to_axes(self, ax: Axes, sequence_idx=0):
        intensity_vals, t_vals, data_timestamps = self._get_intensity(sequence_idx)
        ax.plot(t_vals, intensity_vals, label='hawkes')
        self._plot_data_timestamps(data_timestamps, ax)
        ax.set_title('Hawkes True Model Intensity')


class RMTPPPlotter(BasePlotter):
    def __init__(self, data_model_sandbox: RMTPPDataModelSandBox, plotting_params: PlottingParams):
        super().__init__(plotting_params)
        self.data_model_sandbox = data_model_sandbox

    def _get_intensity(self, sequence_idx):
        """"""
        hidden_seq, data_timestamps = self.data_model_sandbox.setup(sequence_idx)  # # (T, BS=1, h_dim), (T, BS=1, 1)

        log_intensity, evaluated_timestamps = self.data_model_sandbox.get_intensity_over_grid(hidden_seq,
                                                                                              data_timestamps)
        intensity_np = log_intensity.exp().detach().cpu().numpy().flatten()
        evaluated_timestamps_np = evaluated_timestamps.cpu().numpy().flatten()
        data_timestamps_np = data_timestamps.cpu().numpy().flatten()
        return intensity_np, evaluated_timestamps_np, data_timestamps_np

    def plot_intensity_to_axes(self, ax: Axes, sequence_idx=0):
        intensity_vals, t_vals, data_timestamps = self._get_intensity(sequence_idx)
        ax.plot(t_vals, intensity_vals, label='rmtpp')
        ax.scatter(data_timestamps, [min(intensity_vals)] * len(data_timestamps), s=1)
        self._plot_data_timestamps(data_timestamps, ax)
        ax.set_title('RMTPP Intensity')