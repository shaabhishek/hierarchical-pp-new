from argparse import Namespace

import numpy as np
import torch

from data_model_sandbox import get_argparse_parser_params
from engine import Engine
from utils.helper import pretty_print_table

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seeds(seed_value: int):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)


def print_input_params(params: Namespace):
    params_dict = vars(params)
    pretty_print_table('Parameter', 'Value')
    for param, value in params_dict.items():
        pretty_print_table(param, value)


def pipeline_engine(params: Namespace):
    engine = Engine(params, params.model_filename)
    print_input_params(params)
    engine.run()


if __name__ == '__main__':
    # rmtpp_hawkes_params = get_argparse_parser_params('rmtpp', 'simulated_hawkes')
    # set_seeds(rmtpp_hawkes_params.seed)
    # command_line_params = rmtpp_hawkes_params

    # model1_hawkes_params = get_argparse_parser_params('model1', 'simulated_hawkes')
    # set_seeds(model1_hawkes_params.seed)
    # command_line_params = model1_hawkes_params

    command_line_params = get_argparse_parser_params()
    set_seeds(command_line_params.seed)

    pipeline_engine(command_line_params)
