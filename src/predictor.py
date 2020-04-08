from parameters import PredictionParams


class Predictor:
    def __init__(self, prediction_params: PredictionParams):
        self.predictions_save_path = prediction_params.get_logs_file_path()
