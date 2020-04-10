from argparse import Namespace

from matplotlib import pyplot as plt

from data_model_sandbox import RMTPPDataModelSandBox, HawkesProcessDataModelSandBox, get_argparse_parser_params
from hyperparameters import RMTPPHyperparams, HawkesHyperparams
from main import Engine
from parameters import DataModelParams, PlottingParams, DataParams
from plotting import RMTPPPlotter, HawkesPlotter, SavePlotMixin
from utils.data_loader import SingleSequenceDSetCategorical, DLoaderCategorical


class SimulatedModelTestMixin:
    def test_hawkes_true_model(self, params: Namespace):
        assert isinstance(self, SimTrueTestPlot)
        data_params = DataParams(params=params, split_name='train')
        model_hyperparams = HawkesHyperparams(**{'lambda_0': .2, 'alpha': .8, 'beta': 1.})

        data_model_sandbox = HawkesProcessDataModelSandBox(data_params, model_hyperparams)
        hawkes_plotter = HawkesPlotter(data_model_sandbox, self.plotting_params)

        hawkes_plotter.plot_intensity_to_axes(self.ax, sequence_idx=0)
        # hawkes_plotter.save_plot_and_close_fig(fig, "test_hawkes_true_model_idx0")


class NNModelTestMixin:
    def test_rmtpp(self, params: Namespace, model_file_identifier):
        assert isinstance(self, SimTrueTestPlot)
        data_model_params = DataModelParams(
            params=params,
            model_file_identifier=model_file_identifier,
            split_name='train'
        )
        model_hyperparams = RMTPPHyperparams(params)

        data_model_sandbox = RMTPPDataModelSandBox(data_model_params, model_hyperparams)
        plotter = RMTPPPlotter(data_model_sandbox, self.plotting_params)

        plotter.plot_intensity_to_axes(self.ax, sequence_idx=0)
        # plotter.save_plot_and_close_fig(fig, "test_hawkes_rmtpp_idx0")


class SimTrueTestPlot(NNModelTestMixin, SimulatedModelTestMixin, SavePlotMixin):
    def __init__(self, params):
        self.params = params
        self.plotting_params = PlottingParams(params)
        self._figure_dir = self.plotting_params.get_plotting_dir()
        self.fig, self.ax = plt.subplots(1, 1)

    def make_plots(self, filename):
        self.test_rmtpp(self.params, "singletrained_g1_do0.5_b16_h256_l20.0_l20_gn10.0_lr0.001_c10_s1_tlintensity_ai40")
        self.test_hawkes_true_model(self.params)
        self.ax.legend()
        self.save_plot_and_close_fig(self.fig, filename)


def test_single_sequence_training():
    rmtpp_hawkes_params = get_argparse_parser_params('rmtpp', 'simulated_hawkes')
    import pdb; pdb.set_trace()
    engine = Engine(rmtpp_hawkes_params, "singletrained_g1_do0.5_b16_h256_l20.0_l20_gn10.0_lr0.001_c10_s1_tlintensity_ai40")
    data_model_params = DataModelParams(
        params=rmtpp_hawkes_params,
        split_name='train'
    )

    def custom_dataloader():
        data_path = data_model_params.get_data_file_path()
        dataset = SingleSequenceDSetCategorical(data_path, sequence_idx=0)
        loader = DLoaderCategorical(dataset, bs=1)
        return loader

    # train_dataloader = custom_dataloader()
    # engine.train_val_runner = TrainValRunner(engine.trainer_params, train_dataloader,
    #                                          engine.train_val_runner.valid_dataloader, engine.logger)
    engine.run()


def rmtpp_simulated_intensity_plot():
    rmtpp_hawkes_params = get_argparse_parser_params('rmtpp', 'simulated_hawkes')
    SimTrueTestPlot(rmtpp_hawkes_params).make_plots("singletrained_rmtpp_hawkes_comparison_idx0")


if __name__ == "__main__":
    # rmtpp_simulated_intensity_plot()
    test_single_sequence_training()
