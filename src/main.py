from argparse import Namespace

import numpy as np
import torch

from data_model_sandbox import load_dataloader_from_params, get_argparse_parser_params
from hyperparameters import ModelHyperparams, OptimizerHyperparams
from parameters import DataModelParams, LoggingParams, ModelFileParams, PredictionParams, TrainerParams, TestingParams
from run import TrainValRunner, TestRunner
from utils.logger import Logger

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
    for param, value in params_dict:
        _print(param, value)


class Engine:
    def __init__(self, params, model_file_identifier=None):
        self.params = params
        model_file_params = ModelFileParams(params)
        model_hyperparams = ModelHyperparams(params)

        self.logger = self.make_logger(params, model_file_params)

        if self.is_training():
            optimizer_hyperparams = OptimizerHyperparams(params)
            train_data_model_params = DataModelParams(params=params, split_name='train')
            valid_data_model_params = DataModelParams(params=params, split_name='valid')
            self.trainer_params = TrainerParams(params, train_data_model_params, model_hyperparams,
                                                model_file_params, optimizer_hyperparams)
            self.train_val_runner = self.make_train_runner(train_data_model_params, valid_data_model_params,
                                                           self.trainer_params,
                                                           self.logger)

        if self.is_testing():
            test_data_model_params = DataModelParams(params=params,
                                                     model_file_identifier=model_file_identifier,
                                                     # model_file_identifier=model_file_params.get_model_file_identifier(),
                                                     split_name='test')
            prediction_params = PredictionParams(params, model_file_params)
            self.testing_params = TestingParams(params, test_data_model_params, model_hyperparams, prediction_params,
                                                model_file_params)

    def train(self):
        self.train_val_runner.train_dataset()

    def test(self):
        self.test_runner = self.make_test_runner(self.testing_params, self.logger)
        self.test_runner.test_dataset()

    def is_training(self):
        return not self.params.skiptrain

    def is_testing(self):
        return not self.params.skiptest

    def print_dataset_details(self):
        if self.is_training():
            print("Train data length", len(self.train_val_runner.train_dataloader.dataset))
            print("Validation data length", len(self.train_val_runner.valid_dataloader.dataset))
        if self.is_testing():
            print("Test data length", len(self.test_runner.test_dataloader.dataset))

    @staticmethod
    def make_logger(params, model_file_params):
        logging_params = LoggingParams(params=params, model_file_params=model_file_params)
        logger = Logger(logging_params, marker_type=params.marker_type)
        return logger

    @staticmethod
    def make_train_runner(train_data_model_params, valid_data_model_params, trainer_params, logger):
        train_dataloader = load_dataloader_from_params(train_data_model_params)
        valid_dataloader = load_dataloader_from_params(valid_data_model_params)
        train_data_model_params.create_and_set_filename()
        train_val_runner = TrainValRunner(trainer_params, train_dataloader, valid_dataloader, logger)
        return train_val_runner

    @staticmethod
    def make_test_runner(testing_params, logger):
        test_dataloader = load_dataloader_from_params(testing_params.data_model_params)
        test_runner = TestRunner(testing_params, test_dataloader, logger)
        return test_runner

    def run(self):
        if self.is_training():
            self.train()
        if self.is_testing():
            self.test()


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
