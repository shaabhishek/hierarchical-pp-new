from argparse import Namespace

import numpy as np
import torch

from data_model_sandbox import get_argparse_parser_params
from engine import Engine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seeds(seed_value: int):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)


def print_input_params(params: Namespace):
    def _print(a, b):
        print(f"{a:>20}", end='')
        print(f"{b:>20}", end='')
        print()

    params_dict = vars(params)
    _print('Parameter', 'Value')
    for param, value in params_dict.items():
        _print(param, value)


def pipeline_engine(params: Namespace):
    engine = Engine(params)
    print_input_params(params)
    engine.run()


if __name__ == '__main__':
    # rmtpp_hawkes_params = get_argparse_parser_params('rmtpp', 'simulated_hawkes')
    # set_seeds(rmtpp_hawkes_params.seed)

    command_line_params = get_argparse_parser_params()
    set_seeds(command_line_params.seed)

    pipeline_engine(command_line_params)
