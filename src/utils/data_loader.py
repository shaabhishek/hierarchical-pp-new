import pickle

def load_data(data_path):
    # Data is stored as a dict with keys 'x' and 't' into a file with path='path_data'
    # The values of the keys are x_data and t_data:
    # x_data: list of length num_data_train, each element is numpy array of shape T_i x marker_dim
    # t_data: list of length num_data_train, each element is numpy array of shape T_i x 3
    with open(data_path, 'rb') as handle:
        data = pickle.load(handle)
    return data['x'], data['t']

if __name__ == "__main__":
    pass
    # load_model()