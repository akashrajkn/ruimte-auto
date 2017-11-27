import os
import torch
import numpy as np

import torch.utils.data as data_utils


def read_dataset(path=None):
    '''
    Reads csv data and Converts it into vector.
    '''

    with open(path, newline='') as f:
        data_source = f.read()

    headers = []
    features = []
    targets = []
    firstLine = True

    for line in data_source.splitlines():
        row = line.split(',')
        if firstLine:
            # first line
            headers = row
            firstLine = False
            continue

        # FIXME: missing values are ignored here
        if len(row) < 25:
            continue

        targets.append([float(i) for i in row[0:3]])
        features.append([float(i) for i in row[3:25]])

    # TODO: Return headers also,
    #       Possible return type, return feature_tuple, targets_tuple.
    #       Each tuple = (headers, data)
    return features, targets


def create_torch_dataset(path=None):
    '''
    Create a torch dataset from path
    '''

    features = []
    targets = []

    # If path is not specified use, the default dataset
    if path is None:
        realpath = os.path.dirname(os.path.realpath(__file__))
        data_folder = realpath + '/../../data/'
        for data_file in os.listdir(data_folder):
            if data_file.endswith('.csv'):
                temp_features, temp_targets = read_dataset(data_folder + data_file)

                features += temp_features
                targets += temp_targets

    # 80% data is train data
    # num_train = int(len(features) * 0.8)
    # All available data is used for training
    num_train = int(len(features))

    features_train = np.array(features[:num_train], dtype=np.float32)
    targets_train = np.array(targets[:num_train], dtype=np.float32)

    features_test = np.array(features[num_train:], dtype=np.float32)
    targets_test = np.array(targets[num_train:], dtype=np.float32)

    return features_train, targets_train, features_test, targets_test

    # features_train = torch.FloatTensor(features[:num_train])
    # targets_train = torch.FloatTensor(targets[:num_train])
    # features_test = torch.FloatTensor(features[num_train:])
    # targets_test = torch.FloatTensor(targets[num_train:])
    #
    # train_data = data_utils.TensorDataset(features_train, targets_train)
    # train_loader = data_utils.DataLoader(train_data, shuffle=True)
    #
    # test_data = data_utils.TensorDataset(features_test, targets_test)
    # test_loader = data_utils.DataLoader(test_data)
    #
    # return train_loader, test_loader


if __name__ == '__main__':
    create_torch_dataset()
