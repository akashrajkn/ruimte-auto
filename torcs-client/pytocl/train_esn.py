import os
import sys
import dill as pickle
import numpy as np

from preprocess import read_dataset
import echo_state_network

sys.modules['echo_state_network'] = echo_state_network


def squash(a):
    a = 0.49*a/max(a) # squash to values between 0.01 and 0.99
    a = a + .5
    return a


def split_data(data, targets):
    n_train = int((len(data) * 0.6))
    n_cv = int((len(data) * 0.2))
    n_test = len(data) - n_cv - n_train

    # data is not shuffled, because it is supposed to be sequential
    train_data = data[:n_train]
    train_targets = targets[:n_train]

    cv_data = data[n_train:n_train + n_cv]
    cv_targets = targets[n_train:n_train + n_cv]

    test_data = data[n_train + n_cv:]
    test_targets = targets[n_train + n_cv:]

    return [train_data, train_targets], [cv_data, cv_targets], [test_data, test_targets]


def prep_data_for_esn(data, targets):
    '''
    adds a bias term to training data
    squashes the domain of the training data between 0 and 1 (excluding 0 and 1)
    '''

    for d in data:
        d.append(1.0)  # add bias term
    data = np.asarray(data)

    # do not squash bias
    for i in range(len(data[0]) - 1):
        data[:, i] = squash(data[:, i])

    targets = np.asarray(targets)
    return split_data(data, targets)


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


best_param = [.7, .4, .1, .2]

rng = np.random.RandomState(42)
best_t = [.85, .35]

spec_rad = best_param[0]
spars = best_param[0]
nos = best_param[0]
tsh = best_param[0]

esn = echo_state_network.ESN(n_inputs = 23,
          n_outputs = 3,
          n_reservoir = 200,
          spectral_radius = spec_rad,
          sparsity = spars,
          noise = nos,
          #input_shift = [0,0],
          #input_scaling = [0.01, 3],
          #teacher_scaling = .8,
          teacher_shift = tsh,
          #out_activation = np.tanh,
          #inverse_out_activation = np.arctanh,
          teacher_forcing = True,
          random_state = rng,
          silent = True,
          BAD = best_t )

n_reservoir = 200
state = np.zeros(n_reservoir)
control = np.zeros(3)

features = []
targets = []

# Read all training data
realpath = os.path.dirname(os.path.realpath(__file__))
data_folder = realpath + '/../../data/'
for data_file in os.listdir(data_folder):
    if data_file.endswith('.csv'):
        temp_features, temp_targets = read_dataset(data_folder + data_file)

        features += temp_features
        targets += temp_targets

all_train, all_cv, all_test = prep_data_for_esn(features, targets)
all_train_data = np.asarray(list(all_train[0]))
all_train_targ = np.asarray(list(all_train[1]))

esn.fit(all_train_data, all_train_targ)

for i in range(len(all_test[0])):
    sens_vec = all_test[0][i, :]
    control, state = esn.race(sens_vec, control, state)

# sample usage
save_object(esn, realpath + '/../../models/esn.pkl')
