from argparse import Namespace

from matplotlib import pyplot as plt

from data_model_sandbox import get_argparse_parser_params
from engine import Engine
from parameters import PlottingParams, DataParams
from plotting import RMTPPPlotter, HawkesPlotter, BasePlot
from run import TrainValRunner
from utils.data_loader import SingleSequenceDSetCategorical, DLoaderCategorical


class SimulatedModelTestMixin:
    def test_hawkes_true_model(self, params: Namespace):
        assert isinstance(self, MultipleModelPlot)
        hawkes_plotter = self._make_plotter(params)

        # hawkes_plotter.plot_intensity_vs_timestamp_to_axes(self.ax, sequence_idx=0)
        hawkes_plotter.plot_intensity_vs_time_index_to_axes(self.ax, sequence_idx=0)


class NNModelTestMixin:
    def test_rmtpp(self, params: Namespace, model_filename):
        assert isinstance(self, MultipleModelPlot)
        plotter = self._make_plotter(model_filename, params)

        # plotter.plot_intensity_vs_timestamp_to_axes(self.ax, sequence_idx=0)
        plotter.plot_intensity_vs_time_index_to_axes(self.ax, sequence_idx=0)

    @staticmethod
    def _make_plotter(model_filename, params):
        plotter = RMTPPPlotter.from_filename(model_filename, params, split_name='train')
        return plotter


class IntensityVsTimeIndexPlotMixin:
    def _make_single_plot(self, plotter):
        assert isinstance(self, MultipleModelPlot)
        assert isinstance(plotter, (RMTPPPlotter, HawkesPlotter))
        plotter.plot_intensity_vs_time_index_to_axes(self.ax, sequence_idx=0)


class MultipleModelPlot(BasePlot):
    def __init__(self, plotting_params, plotter_list):
        super(MultipleModelPlot, self).__init__(plotting_params)
        self.plotter_list = plotter_list
        self.fig, self.ax = plt.subplots(1, 1)

    def plot(self):
        for plotter in self.plotter_list:
            self._make_single_plot(plotter)
        self.ax.legend()

    @classmethod
    def from_params(cls, params, plotter_list):
        plotting_params = PlottingParams(params)
        return cls(plotting_params, plotter_list)


class RMTPPHawkesIntensityTimeIndexPlot(IntensityVsTimeIndexPlotMixin, MultipleModelPlot):
    pass


def test_plot_time_index():
    rmtpp_hawkes_params = get_argparse_parser_params('rmtpp', 'simulated_hawkes')
    plot_object = RMTPPHawkesIntensityTimeIndexPlot.from_params(
        params=rmtpp_hawkes_params,
        plotter_list=[
            RMTPPPlotter.from_filename(
                "stored_g1_do0.5_b16_h256_l20.0_l20_gn10.0_lr0.001_c10_s1_tlintensity_ai40.pt",
                params=rmtpp_hawkes_params,
                split_name="train"
            ),
            HawkesPlotter.from_hyperparams(
                {'lambda_0': .2, 'alpha': .8, 'sigma': 1.},
                params=rmtpp_hawkes_params,
                split_name='train'
            )
        ]
    )
    plot_object.plot()
    plot_object.save_plot_and_close_fig(plot_object.fig, "rmtpp_hawkes_intensity_vs_time_index")


def train_single_sequence():
    rmtpp_hawkes_params = get_argparse_parser_params('rmtpp', 'simulated_hawkes')
    engine = Engine(
        rmtpp_hawkes_params,
        "singletrained_g1_do0.5_b16_h256_l20.0_l20_gn10.0_lr0.001_c10_s1_tlintensity_ai40.pt"
    )
    data_params = DataParams(
        params=rmtpp_hawkes_params,
        split_name='train'
    )

    def custom_dataloader():
        data_path = data_params.get_data_file_path()
        dataset = SingleSequenceDSetCategorical(data_path, sequence_idx=0)
        loader = DLoaderCategorical(dataset, bs=1)
        return loader

    train_dataloader = custom_dataloader()
    engine.train_val_runner = TrainValRunner(engine.trainer_params, train_dataloader,
                                             engine.train_val_runner.valid_dataloader, engine.logger)
    engine.run()


def test_only_model():
    rmtpp_hawkes_params = get_argparse_parser_params('rmtpp', 'simulated_hawkes')
    rmtpp_hawkes_params.skiptrain = True
    engine = Engine(
        rmtpp_hawkes_params,
        "singletrained_g1_do0.5_b16_h256_l20.0_l20_gn10.0_lr0.001_c10_s1_tlintensity_ai40.pt"
    )
    engine.run()


def rmtpp_simulated_intensity_plot():
    rmtpp_hawkes_params = get_argparse_parser_params('rmtpp', 'simulated_hawkes')
    MultipleModelPlot(rmtpp_hawkes_params).make_plots("singletrained_rmtpp_hawkes_comparison_nogrid_idx0")


if __name__ == "__main__":
    # rmtpp_simulated_intensity_plot()
    train_single_sequence()
    # test_only_model()
    # test_plot_time_index()
