from argparse import Namespace

import numpy as np
import torch

from data_model_sandbox import load_dataloader_from_params, get_argparse_parser_params
from hyperparameters import ModelHyperparams, OptimizerHyperparams
from parameters import DataModelParams, LoggingParams, ModelFileParams, PredictionParams, _augment_params, setup_parser, \
    TrainerParams, TestingParams
from run import TrainValRunner, TestRunner
from utils.data_loader import get_dataloader
from utils.logger import Logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seeds(seed_value: int):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)


def print_input_params(params):
    d = vars(params)
    for key in d:
        print('\t', key, '\t', d[key])


def is_training(params: Namespace):
    return not params.test


def pipeline_engine(params: Namespace):
    model_file_params = ModelFileParams(params)
    model_hyperparams = ModelHyperparams(params)
    logging_params = LoggingParams(params=params, model_file_params=model_file_params)
    logger = Logger(logging_params, marker_type=params.marker_type)
    if is_training(params):
        train_data_model_params = DataModelParams(params=params, split_name='train')
        valid_data_model_params = DataModelParams(params=params, split_name='valid')
        optimizer_hyperparams = OptimizerHyperparams(params)
        trainer_hyperparams = TrainerParams(params, train_data_model_params, model_hyperparams, model_file_params,
                                            optimizer_hyperparams)

        print_input_params(params)

        # Data should reside in this path for all datasets. Ideally 5 cross fold validation.

        train_dataloader = load_dataloader_from_params(train_data_model_params)
        valid_dataloader = load_dataloader_from_params(valid_data_model_params)

        print("\n")
        print("train data length", len(train_dataloader.dataset))
        print("valid data length", len(valid_dataloader.dataset))
        print("\n")

        train_runner = TrainValRunner(trainer_hyperparams, train_dataloader, valid_dataloader, logger)
        train_runner.train_dataset()
        # train_one_dataset(params, file_name, train_dataloader, valid_dataloader, logger)
        if params.train_test:
            test_data_model_params = DataModelParams(params=params,
                                                     model_file_identifier=model_file_params.get_model_file_identifier(),
                                                     split_name='test')
            prediction_params = PredictionParams(params, model_file_params)
            testing_hyperparams = TestingParams(params, test_data_model_params, model_hyperparams, prediction_params,
                                                model_file_params)

            test_dataloader = load_dataloader_from_params(test_data_model_params)
            test_runner = TestRunner(testing_hyperparams, test_dataloader, logger)
            test_runner.test_dataset()
            # test_data_path = params.data_dir + "/" + params.data_name + '_' + str(params.cv_idx) + "_test.pkl"
            # test_dataloader = get_dataloader(test_data_path, params.marker_type, params.batch_size)
            # test_one_dataset(params, file_name, test_dataloader, logger, save=True)
    else:

        test_data_model_params = DataModelParams(params=params,
                                                 model_file_identifier=model_file_params.get_model_file_identifier(),
                                                 split_name='test')
        prediction_params = PredictionParams(params, model_file_params)
        testing_hyperparams = TestingParams(params, test_data_model_params,
                                            model_hyperparams, prediction_params, model_file_params)

        best_epoch = params.best_epoch  # TODO: Make the test runner aware of best epoch
        test_dataloader = load_dataloader_from_params(test_data_model_params)
        test_runner = TestRunner(testing_hyperparams, test_dataloader, logger)
        # test_data_path = params.data_dir + "/" + params.data_name + '_' + str(params.cv_idx) + "_test.pkl"
        # test_dataloader = get_dataloader(test_data_path, params.marker_type, params.batch_size)
        test_runner.test_dataset()
        # file_name = ''
        # for item_ in file_name_identifier:
        #     file_name = file_name + item_[0] + str(item_[1])
        #
        # test_one_dataset(params, file_name, test_dataloader, best_epoch)


if __name__ == '__main__':
    rmtpp_hawkes_params = get_argparse_parser_params('rmtpp', 'simulated_hawkes')
    # parser = setup_parser()
    # params = parser.parse_args()
    # params = _augment_params(params)

    set_seeds(rmtpp_hawkes_params.seed)

    pipeline_engine(rmtpp_hawkes_params)
