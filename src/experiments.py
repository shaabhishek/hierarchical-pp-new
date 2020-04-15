from data_model_sandbox import get_argparse_parser_params
from engine import Engine
from test_models import plot_model_intensity_vs_time_index


class ExperimentToBeNamed:
    def __init__(self):
        self.model_filenames = [
            "rmtpp_mc_50_g1_do0.5_b16_h128_l20.0_l20_gn1000.0_lr0.01_c10_s1_tlintensity_ai40_20_04_14_18_08_10.pt",
        ]
        self.model_labels = ["rmtpp_mc_expt50",]

    def make_plots(self):
        for model_filename, model_label in zip(self.model_filenames, self.model_labels):
            plot_model_intensity_vs_time_index(model_filename, 'rmtpp', 'simulated_hawkes', label=model_label)

    def inspect_mc_models(self):
        rmtpp_hawkes_params = get_argparse_parser_params('rmtpp', 'simulated_hawkes')
        engines = [Engine(rmtpp_hawkes_params, model_filename) for model_filename in self.model_filenames]
        import pdb; pdb.set_trace()


if __name__ == '__main__':
    ExperimentToBeNamed().make_plots()
