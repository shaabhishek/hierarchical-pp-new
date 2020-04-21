class EpochMetrics:
    def __init__(self, marker_type: str):
        self.marker_type = marker_type
        self.total_data_points = 0.
        self.total_loss = 0.

        self.marker_nll = 0.
        self.time_nll = 0.

        self.time_mse = 0.
        self.time_mse_count = 0.

        if self.marker_type == 'real':
            self.marker_mse = 0.
            self.marker_mse_count = 0.
        else:
            self.marker_acc = 0.
            self.marker_acc_count = 0.

    def update_batch_metrics(self, meta_info: dict):
        self._update_total_loss_batch(meta_info)
        self._update_marker_time_likelihood_batch(meta_info)

        self._update_time_mse_batch(meta_info)

        if self.marker_type == 'real':
            self._update_marker_mse_batch(meta_info)
        else:
            self._update_marker_accuracy_batch(meta_info)

    def get_reduced_metrics(self, num_sequences):
        """

        :rtype: dict
        """
        time_rmse = self._reduce_time_rmse()
        total_loss = self._reduce_mean(num_sequences)
        marker_nll, time_nll = self._reduce_marker_time_likelihood(num_sequences)
        if self.marker_type == 'real':
            marker_rmse = self._reduce_marker_mse()
            marker_accuracy = None
            marker_auc = None
        else:
            marker_accuracy = self._reduce_marker_accuracy()
            marker_auc = None
            marker_rmse = None
        reduced_metric_dict = {'loss': total_loss, 'time_rmse': time_rmse, 'accuracy': marker_accuracy,
                               'auc': marker_auc, 'marker_rmse': marker_rmse, 'marker_nll': marker_nll,
                               'time_nll': time_nll}
        return reduced_metric_dict

    def _reduce_marker_accuracy(self):
        marker_accuracy = self.marker_acc / self.marker_acc_count
        return marker_accuracy

    def _reduce_marker_mse(self):
        marker_rmse = (self.marker_mse / self.marker_mse_count) ** 0.5
        return marker_rmse

    def _reduce_mean(self, N):
        total_loss = self.total_loss / N
        return total_loss

    def _reduce_time_rmse(self):
        time_rmse = (self.time_mse / self.time_mse_count) ** 0.5
        return time_rmse

    def _reduce_marker_time_likelihood(self, N):
        marker_nll = self._reduce_mean(N)
        time_nll = self._reduce_mean(N)
        return marker_nll, time_nll

    def _update_total_loss_batch(self, meta_info):
        self.total_loss += meta_info['true_nll'].numpy()

    def _update_marker_accuracy_batch(self, meta_info):
        self.marker_acc += meta_info["marker_acc"]
        self.marker_acc_count += meta_info["marker_acc_count"]

    def _update_marker_mse_batch(self, meta_info):
        self.marker_mse += meta_info["marker_mse"]
        self.marker_mse_count = meta_info["marker_mse_count"]

    def _update_time_mse_batch(self, meta_info):
        self.time_mse += meta_info["time_mse"]
        self.time_mse_count += meta_info["time_mse_count"]

    def _update_marker_time_likelihood_batch(self, meta_info):
        self.marker_nll += meta_info['marker_nll'].numpy()
        self.time_nll += meta_info['time_nll'].numpy()
