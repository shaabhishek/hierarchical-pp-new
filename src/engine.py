from data_model_sandbox import load_dataloader_from_params
from hyperparameters import ModelHyperparams, OptimizerHyperparams
from parameters import ModelFileParams, TrainerParams, PredictionParams, TestingParams, DataModelParams, LoggingParams
from run import TrainValRunner, TestRunner
from utils.logger import Logger


class Engine:
    def __init__(self, params, model_filename=None):
        self.params = params
        model_file_params = ModelFileParams(params)
        model_hyperparams = ModelHyperparams(params)

        self.logger = self.make_logger(params, model_file_params)

        if self.is_training():
            optimizer_hyperparams = OptimizerHyperparams(params)
            train_data_model_params, valid_data_model_params, model_filename = self.get_train_valid_data_model_params(
                params, model_filename, model_file_params)
            self.trainer_params = TrainerParams(params, train_data_model_params, model_hyperparams,
                                                model_file_params, optimizer_hyperparams)
            self.train_val_runner = self.make_train_runner(train_data_model_params, valid_data_model_params,
                                                           self.trainer_params,
                                                           self.logger)

        if self.is_testing():
            test_data_model_params = self.get_test_data_model_params(params, model_filename)
            prediction_params = PredictionParams(params, model_file_params)
            self.testing_params = TestingParams(params, test_data_model_params, model_hyperparams, prediction_params,
                                                model_file_params)

    @staticmethod
    def get_test_data_model_params(params, model_filename):
        if model_filename is None:
            raise ValueError("Testing mode requires the model file path")
        test_data_model_params = DataModelParams(params=params, model_filename=model_filename, split_name='test')
        return test_data_model_params

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
    def get_train_valid_data_model_params(params, model_filename, model_file_params):
        if model_filename is not None:
            train_data_model_params = DataModelParams(params=params, model_filename=model_filename,
                                                      split_name='train')
        else:
            train_data_model_params = DataModelParams.from_identifier(params,
                                                                      model_file_params.get_model_file_identifier(),
                                                                      'train')
            model_filename = train_data_model_params.get_model_filename()
        valid_data_model_params = DataModelParams(params=params, model_filename=None, split_name='valid')
        return train_data_model_params, valid_data_model_params, model_filename

    @staticmethod
    def make_logger(params, model_file_params):
        logging_params = LoggingParams(params=params, model_file_params=model_file_params)
        logger = Logger(logging_params, marker_type=params.marker_type)
        return logger

    @staticmethod
    def make_train_runner(train_data_model_params, valid_data_model_params, trainer_params, logger):
        train_dataloader = load_dataloader_from_params(train_data_model_params)
        valid_dataloader = load_dataloader_from_params(valid_data_model_params)
        train_val_runner = TrainValRunner(trainer_params, train_dataloader, valid_dataloader, logger)
        return train_val_runner

    # @staticmethod
    # def get_data_model_params(params, model_filename, split_name):
    #
    #     train_data_model_params.create_and_set_filename()

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