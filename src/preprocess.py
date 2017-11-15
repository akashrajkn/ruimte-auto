import os


def read_dataset(path=None):
    '''
    Reads csv data and Converts it into vector.
    '''
    # If path is not specified use, the default dataset
    if path is None:
        path = os.path.dirname(__file__) + '/../data/aalborg.csv'

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
        targets.append([float(i) for i in row[0:3]])
        features.append([float(i) for i in row[3:]])

    # TODO: Return headers also,
    #       Possible return type, return feature_tuple, targets_tuple.
    #       Each tuple = (headers, data)
    return features, targets


if __name__ == '__main__':
    read_dataset()
