from data_model_sandbox import get_argparse_parser_params
from engine import Engine
from parameters import DataParams
from plotting import HawkesPlotter, MultipleModelPlot, Model1Plotter, RMTPPPlotter
from run import TrainValRunner
from utils.data_loader import SingleSequenceDSetCategorical, DLoaderCategorical

from collections import namedtuple

PlotterData = namedtuple('PlotterData', ['plotter_class', 'model_filename'])


def plot_model_intensity_vs_time_index(model_name, dataset_name, plotter_data_list, label, sequence_idx=0):
    model_dataset_params = get_argparse_parser_params(model_name, dataset_name)
    plotter_list = [
        HawkesPlotter.from_hyperparams(
                {'lambda_0': 0, 'alpha': .8, 'sigma': 1.},
                params=model_dataset_params,
                split_name='train'
            )]
    for plotter_data in plotter_data_list:
        plotter_list.append(
            plotter_data.plotter_class.from_filename(
                model_filename=plotter_data.model_filename,
                params=model_dataset_params,
                split_name="train"
            ))
    plot_object = MultipleModelPlot.from_params(
        params=model_dataset_params,
        plotter_list=plotter_list
    )
    plot_object.plot("Intensity vs Time Index", sequence_idx)
    plot_object.save_plot_and_close_fig(plot_object.fig, f"{label}_rmtpp_hawkes_intensity_vs_time_index")


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
                                             engine.train_val_runner.valid_dataloader, engine.logger,
                                             is_loading_previous_model=False)
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
    MultipleModelPlot.from_params(rmtpp_hawkes_params).plot("singletrained_rmtpp_hawkes_comparison_nogrid_idx0")


if __name__ == "__main__":
    # rmtpp_simulated_intensity_plot()
    # train_single_sequence()
    # test_only_model()
    plot_model_intensity_vs_time_index(
        "rmtpp",
        "simulated_hawkes",
        plotter_data_list=[
            # PlotterData(
            #     RMTPPPlotter,
            #     "rmtpp_mc_50_g1_do0.5_b16_h128_l20.0_l20_gn1000.0_lr0.01_c10_s1_tlintensity_ai40_20_04_14_18_08_10.pt"
            # ),
            # PlotterData(
            #     Model1Plotter,
            #     "test_model_1_g1_do0.5_b32_h256_l20.0_l20_gn10.0_lr0.001_c10_s1_tlintensity_ai40_20_04_20_12_12_20.pt"
            # )
        ],
        # "rmtpp_vs_hawkes",
        # "model1_vs_hawkes",
        label="hawkes", sequence_idx=20
    )
