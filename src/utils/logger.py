from torch.utils.tensorboard import SummaryWriter

from parameters import LoggingParams


class Logger:
    def __init__(self, logging_params: LoggingParams, marker_type: str):
        self.model_name = logging_params.model_name
        self.dataset_name = logging_params.dataset_name

        # NOTE: Only saving the timestamp upto the second
        self.logs_save_path = logging_params.get_logs_file_path()

        self.marker_type = marker_type
        self.metrics = ['loss', 'marker_ll', 'time_ll', 'accuracy', 'time_rmse']
        self.logged_metrics = {}
        self.best_epoch = {}
        self.best_valid_loss = {}
        self.improvement_map = {'loss': 'small', 'marker_ll': 'large', 'time_ll': 'large', 'auc': 'large',
                                'accuracy': 'large', 'marker_rmse': 'small', 'time_rmse': 'small'}

        self.writer = SummaryWriter(log_dir=logging_params.get_tensorboard_log_dir())

        self.init_logged_metrics()

    def init_logged_metrics(self):
        for split in ['train', 'valid', 'test']:
            self.logged_metrics[split] = {}
            for metric_name in self.metrics:
                self.logged_metrics[split][metric_name] = {}

        for metric_name in self.metrics:
            self.best_epoch[metric_name] = None
            self.best_valid_loss[metric_name] = None

    def get_best_epoch(self, metric_name='loss'):
        """
        Returns the best epoch so far with respect to a certain metric
        """
        if self.best_epoch[metric_name] is not None:
            return self.best_epoch[metric_name]
        else:
            return None

    def log_train_epoch(self, epoch_num: int, train_info_dict: dict, valid_info_dict: dict):
        for metric_name in self.metrics:
            # Write to internal state
            self.logged_metrics['train'][metric_name][epoch_num] = train_info_dict.get(metric_name, float('inf'))
            self.logged_metrics['valid'][metric_name][epoch_num] = valid_info_dict.get(metric_name, float('inf'))

            self.write_train_val_metrics_to_tensorboard(epoch_num, metric_name)

            if (self.best_valid_loss[metric_name] is None) or \
                    (self.improvement_map[metric_name] == 'small' and valid_info_dict[metric_name] <
                     self.best_valid_loss[metric_name]) or \
                    (self.improvement_map[metric_name] == 'large' and valid_info_dict[metric_name] >
                     self.best_valid_loss[metric_name]):
                self.best_valid_loss[metric_name] = valid_info_dict[metric_name]
                self.best_epoch[metric_name] = epoch_num

    def write_train_val_metrics_to_tensorboard(self, epoch_num, metric_name):
        for split in ['train', 'valid']:
            self.writer.add_scalar(f"{metric_name}/{split}", self.logged_metrics[split][metric_name][epoch_num],
                                   epoch_num)

    def log_test_epoch(self, epoch_num: int, test_info_dict: dict):
        for metric_name in self.metrics:
            self.logged_metrics['test'][metric_name][epoch_num] = test_info_dict.get(metric_name, float('inf'))

    def print_train_epoch(self, epoch_num: int, train_info_dict: dict, valid_info_dict: dict):
        def _format_line(metric_name, valid_metric_val, train_metric_val):
            return f"Validation {metric_name}: {valid_metric_val:.3f}, \t\t\t Train {metric_name}: {train_metric_val:.3f}"

        print('epoch', epoch_num + 1)
        if self.marker_type == 'categorical':
            print(
                f"Validation Accuracy: {valid_info_dict['accuracy']:.3f}, \t\t\t Train Accuracy: {train_info_dict['accuracy']:.3f}")
        else:
            raise NotImplementedError

        print(_format_line('Loss', valid_info_dict['loss'], train_info_dict['loss']))
        print(_format_line('Marker LL', valid_info_dict['marker_ll'], train_info_dict['marker_ll']))
        print(_format_line('Time LL', valid_info_dict['time_ll'], train_info_dict['time_ll']))
        print(_format_line('Time RMSE', valid_info_dict['time_rmse'], train_info_dict['time_rmse']))

    def print_test_epoch(self, test_info_dict: dict):
        if self.marker_type == 'categorical':
            print(f"Test Accuracy: {test_info_dict['accuracy']:.3f}")
        else:
            raise NotImplementedError
        print(f"Test Loss: {test_info_dict['loss']:.3f}")
        print(f"Test Marker LL: {test_info_dict['marker_ll']:.3f}")
        print(f"Test Time LL: {test_info_dict['time_ll']:.3f}")
        print(f"Test Time RMSE: {test_info_dict['time_rmse']:.3f}")

    def save_logs_to_file(self, split: str):
        """
        Saves the logged_metrics dict to files
        split can only be in {train, valid, test}
        """
        assert (split in ['train', 'valid', 'test'])

        with open(self.logs_save_path, 'a') as fobj:
            for metric_name in self.metrics:
                lines = f"{split}_{metric_name}:\n{str(self.logged_metrics[split][metric_name])}\n\n"
                fobj.write(lines)

        print(f"Saved {split} logs to file: {self.logs_save_path}")


def test():
    config = {
        "marker_type": "categorical",
        "logs_save_path": "/home/abhishekshar/hierarchichal_point_process/src/utils/logging_testbed/dummylogfile"
    }
    logger = Logger(**config)

    from random import random, randint
    def _gen_dummy_info():
        metric_list = ['loss', 'marker_ll', 'time_ll', 'accuracy', 'marker_rmse', 'time_rmse', 'auc']
        dummy_info = {}

        for metric in metric_list:
            dummy_info[metric] = random()
        return dummy_info

    for _ in range(5):
        epoch_num = randint(0, 100)
        dummy_train_info = _gen_dummy_info()
        dummy_val_info = _gen_dummy_info()

        logger.print_train_epoch(epoch_num, dummy_train_info, dummy_val_info)
        logger.log_train_epoch(epoch_num, dummy_train_info, dummy_val_info)

    logger.save_logs_to_file('train')
    logger.save_logs_to_file('valid')

    dummy_test_info = _gen_dummy_info()
    logger.print_test_epoch(dummy_test_info)
    logger.log_test_epoch(logger.get_best_epoch('loss'), dummy_test_info)
    logger.save_logs_to_file('test')


if __name__ == "__main__":
    test()
