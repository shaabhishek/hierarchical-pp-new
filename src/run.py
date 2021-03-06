import torch
from torch.utils.data import DataLoader

from epoch_runner import TrainEpochRunner, ValidEpochRunner, TestEpochRunner
from optimizer_loader import OptimizerLoader
from parameters import TrainerParams, TestingParams
from utils.logger import Logger
from utils.model_loader import ModelLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseRunner:
    def __init__(self, logger: Logger):
        self.logger = logger


class TrainValRunner(BaseRunner):
    def __init__(self, trainer_params: TrainerParams, train_dataloader: DataLoader, valid_dataloader: DataLoader,
                 logger, is_loading_previous_model):
        super(TrainValRunner, self).__init__(logger)
        self.trainer_hyperparams = trainer_params

        self.model_state_path = trainer_params.data_model_params.get_model_state_path()

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.model, model_epoch_num = self.load_model(trainer_params, is_loading_previous_model)
        self.optimizer = self.load_optimizer(trainer_params, self.model, is_loading_previous_model)
        self.starting_epoch_num = 1 if is_loading_previous_model is None else model_epoch_num

        self.train_epoch_runner = TrainEpochRunner(self.model, self.train_dataloader, self.optimizer, logger,
                                                   trainer_params.model_hyperparams.total_anneal_epochs,
                                                   trainer_params.model_hyperparams.grad_max_norm)
        self.valid_epoch_runner = ValidEpochRunner(self.model, self.valid_dataloader, logger)

        self.model.print_model_info()

    @staticmethod
    def load_optimizer(trainer_params, model, is_loading_previous_model):
        if is_loading_previous_model:
            optimizer = OptimizerLoader.from_model_checkpoint(
                trainer_params.data_model_params.get_model_state_path(),
                model, trainer_params.optimizer_hyperparams)
        else:
            optimizer = OptimizerLoader.from_scratch(model, trainer_params.optimizer_hyperparams)
        return optimizer

    @staticmethod
    def load_model(trainer_params, is_loading_previous_model):
        if is_loading_previous_model:
            model, epoch_num = ModelLoader.from_model_checkpoint(trainer_params.data_model_params)
            return model, epoch_num
        else:
            model = ModelLoader(trainer_params.model_hyperparams).model
            return model, 1

    def train_dataset(self, starting_epoch):
        """
        Loss is the ELBO
        Accuracy is for categorical/binary marker,
        AUC is for binary/categorical marker.
        Time RMSE is w.r.t expectation.
        Marker rmse for real marker####
        :param starting_epoch:
        """
        self.training_start_hook()
        for epoch_num in range(starting_epoch, self.trainer_hyperparams.num_training_iterations + starting_epoch):
            train_metrics, valid_metrics = self._run_one_train_and_valid_epoch(epoch_num)

            self.log_epoch_info(epoch_num, train_metrics, valid_metrics)
            if self.is_best_epoch(epoch_num):
                self.checkpoint_model(epoch_num, train_metrics['loss'])
        self.training_end_hook()

    def training_start_hook(self):
        pass

    def training_end_hook(self):
        print(f"Training finished. Best epoch:{self.logger.get_best_epoch(metric_name='loss')}")
        self.save_training_session_logs()

    def _run_one_train_and_valid_epoch(self, epoch_num):
        train_metrics = self.train_epoch_runner.run_epoch(epoch_num)
        valid_metrics = self.valid_epoch_runner.run_epoch(epoch_num)
        return train_metrics, valid_metrics

    def save_training_session_logs(self):
        # End of training: save the logger state (metric values) to file
        for split in ['train', 'valid']:
            self.logger.save_logs_to_file(split)
        print(f"Model saved at {self.model_state_path}")

    def is_best_epoch(self, idx):
        return self.logger.get_best_epoch(metric_name='loss') == idx

    def log_epoch_info(self, epoch_num, train_metrics, valid_metrics):
        # print train and validation metric values
        self.logger.print_train_epoch(epoch_num, train_metrics, valid_metrics)
        # save train and validation metric values
        self.logger.log_train_epoch(epoch_num, train_metrics, valid_metrics)
        self.logger.write_train_val_metrics_to_tensorboard(epoch_num)

    def checkpoint_model(self, epoch_num, loss_value):
        state = {
            'epoch': epoch_num,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss_value,
            'model_hyperparams': self.trainer_hyperparams.model_hyperparams
        }
        torch.save(state, self.model_state_path)


class TestRunner(BaseRunner):
    def __init__(self, testing_params: TestingParams, test_dataloader: DataLoader, logger: Logger):
        super(TestRunner, self).__init__(logger)
        self.test_dataloader = test_dataloader

        self.model_state_path = testing_params.data_model_params.get_model_state_path()

        self.model, _ = ModelLoader.from_model_checkpoint(testing_params.data_model_params)

        self.test_epoch_runner = TestEpochRunner(self.model, self.test_dataloader, logger)

        # self.predictions_saver = Predictor(testing_params.prediction_params)

    def test_dataset(self):
        best_epoch_num = self.testing_start_hook()

        test_info = self._run_one_epoch(best_epoch_num)
        self.testing_end_hook(best_epoch_num, test_info)

    def testing_start_hook(self):
        print(f"\n{'Start testing':*^80}\n")
        best_epoch_num = self.logger.get_best_epoch(metric_name='loss')
        if best_epoch_num is None:
            best_epoch_num = -1
        return best_epoch_num

    def testing_end_hook(self, best_epoch_num, test_info):
        self.log(best_epoch_num, test_info)

    def _run_one_epoch(self, epoch_num):
        test_metrics = self.test_epoch_runner.run_epoch(epoch_num)
        return test_metrics

    def log(self, best_epoch_num, test_info):
        # Print the test metric info
        self.logger.print_test_epoch(test_info)
        # Log the test metric info
        self.logger.log_test_epoch(best_epoch_num, test_info)
        # End of epoch: save the logger state (metric values) to file
        self.logger.save_logs_to_file('test')
