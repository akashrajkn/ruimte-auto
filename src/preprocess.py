import os
import torch

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
        if len(row) < 24:
            continue

        targets.append([float(i) for i in row[0:3]])
        features.append([float(i) for i in row[3:24]])

    # TODO: Return headers also,
    #       Possible return type, return feature_tuple, targets_tuple.
    #       Each tuple = (headers, data)
    return features, targets


def create_torch_dataset(path=None):
    '''
    Create a torch dataset from path
    '''
    # If path is not specified use, the default dataset
    if path is None:
        path = path = os.path.dirname(__file__) + '/../data/aalborg.csv'

    features, targets = read_dataset(path)

    # 80% data is train data
    num_train = int(len(features) * 0.8)

    features_train = torch.FloatTensor(features[:num_train])
    targets_train = torch.FloatTensor(targets[:num_train])
    features_test = torch.FloatTensor(features[num_train:])
    targets_test = torch.FloatTensor(targets[num_train:])

    train_data = data_utils.TensorDataset(features_train, targets_train)
    train_loader = data_utils.DataLoader(train_data, shuffle=True)

    test_data = data_utils.TensorDataset(features_test, targets_test)
    test_loader = data_utils.DataLoader(test_data)

    return train_loader, test_loader


if __name__ == '__main__':
    create_torch_dataset()
