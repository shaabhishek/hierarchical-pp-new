from data_model_sandbox import load_dataloader_from_params
from hyperparameters import OptimizerHyperparams, BaseModelHyperParams
from parameters import ModelFileParams, TrainerParams, PredictionParams, TestingParams, DataModelParams, LoggingParams
from run import TrainValRunner, TestRunner
from utils.logger import Logger


class Engine:
    def __init__(self, params, model_filename=None):
        self.params = params
        self.model_file_params = ModelFileParams(params)
        self.model_hyperparams = BaseModelHyperParams.from_params(params)
        self.model_filename = model_filename

        self.logger = self._make_logger(self.params, self.model_file_params, self.model_filename)

    def run(self):
        try:
            if self._is_training():
                self.train()
            if self._is_testing():
                self.test()
        finally:
            self.logger.writer.close()

    def train(self):
        self._setup_train(self.params, self.model_file_params, self.model_hyperparams, self.model_filename)
        self.train_val_runner.train_dataset(self.train_val_runner.starting_epoch_num)
        self.model_filename = self.trainer_params.data_model_params.get_model_filename()

    def test(self):
        self._setup_test(self.params, self.model_file_params, self.model_hyperparams, self.model_filename)
        self.test_runner = self.make_test_runner(self.testing_params, self.logger)
        self.test_runner.test_dataset()

    def _setup_train(self, params, model_file_params, model_hyperparams, model_filename):
        optimizer_hyperparams = OptimizerHyperparams(params)
        train_data_model_params, valid_data_model_params, model_filename = self.get_train_valid_data_model_params(
            params, model_filename, model_file_params)
        self.trainer_params = TrainerParams(params, train_data_model_params, model_hyperparams,
                                            model_file_params, optimizer_hyperparams)
        self.train_val_runner = self.make_train_runner(train_data_model_params, valid_data_model_params,
                                                       self.trainer_params, self.logger, self._is_loading_previous_model())
        if self._is_loading_previous_model():
            assert self.trainer_params.data_model_params.get_model_state_path().exists()

    def _setup_test(self, params, model_file_params, model_hyperparams, model_filename):
        test_data_model_params = self.get_test_data_model_params(params, model_filename)
        prediction_params = PredictionParams(params, model_file_params)
        self.testing_params = TestingParams(params, test_data_model_params, model_hyperparams, prediction_params,
                                            model_file_params)
        assert self.testing_params.data_model_params.get_model_state_path().exists()

    def _is_loading_previous_model(self) -> bool:
        return self.model_filename is not None

    def get_test_data_model_params(self, params, model_filename):
        if not self._is_loading_previous_model():
            raise ValueError("Can't test model without providing correct model file path")
        test_data_model_params = DataModelParams(params=params, model_filename=model_filename, split_name='test')
        assert test_data_model_params.get_model_state_path().exists()
        return test_data_model_params

    def _is_training(self):
        return not self.params.skiptrain

    def _is_testing(self):
        return not self.params.skiptest

    def __repr__(self):
        output_str = ""
        if self._is_training():
            output_str += f"\nFor Training:"
            if self._is_loading_previous_model():
                output_str += f"\nLoading model from file: {self.model_filename}"
            output_str += f"\nTrain data length: {len(self.train_val_runner.train_dataloader.dataset):20}"
            output_str += f"\nValidation data length: {len(self.train_val_runner.valid_dataloader.dataset):20}"

        if self._is_testing():
            output_str += f"\nFor Testing:"
            if self._is_loading_previous_model():
                output_str += f"\nLoading model from file: {self.model_filename}"
            output_str += f"\nTest data length: {len(self.test_runner.test_dataloader.dataset):20}"

        return output_str

    @staticmethod
    def get_train_valid_data_model_params(params, model_filename, model_file_params):
        if model_filename is not None:
            train_data_model_params = DataModelParams(params=params, model_filename=model_filename,
                                                      split_name='train')
            assert train_data_model_params.get_model_state_path().exists()
        else:
            train_data_model_params = DataModelParams.from_identifier(params,
                                                                      model_file_params.get_model_file_identifier(),
                                                                      'train')
            model_filename = train_data_model_params.get_model_filename()
        valid_data_model_params = DataModelParams(params=params, model_filename=None, split_name='valid')
        return train_data_model_params, valid_data_model_params, model_filename

    @staticmethod
    def _make_logger(params, model_file_params, model_filename):
        if model_filename is not None:
            logging_params = LoggingParams.from_model_filename(model_filename, params, model_file_params)
        else:
            logging_params = LoggingParams(params, model_file_params)
        logger = Logger(logging_params, marker_type=params.marker_type)
        return logger

    @staticmethod
    def make_train_runner(train_data_model_params, valid_data_model_params, trainer_params, logger,
                          is_loading_previous_model):
        train_dataloader = load_dataloader_from_params(train_data_model_params)
        valid_dataloader = load_dataloader_from_params(valid_data_model_params)
        train_val_runner = TrainValRunner(trainer_params, train_dataloader, valid_dataloader, logger, is_loading_previous_model)
        return train_val_runner

    @staticmethod
    def make_test_runner(testing_params, logger):
        test_dataloader = load_dataloader_from_params(testing_params.data_model_params)
        test_runner = TestRunner(testing_params, test_dataloader, logger)
        return test_runner