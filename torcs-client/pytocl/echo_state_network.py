import numpy as np
import os

import pytocl.preprocess as preprocess
# import preprocess

# This cell contains PyESN
# ADJUSTED FOR BRAM AKASH DMITRII (BAD) 17 nov 2017

def squash(a):
    a = 0.49*a/max(a) # squash to values between 0.01 and 0.99
    a = a + .5
    return a

def split_data(data, targets):
    n_data = len(data)

    n_train = int((len(data) * 0.6))
    n_cv = int((len(data) * 0.2))
    n_test = n_data - n_cv - n_train

    # data is not shuffled, because it is supposed to be sequential

    train_data = data[:n_train]
    train_targets = targets[:n_train]

    cv_data = data[n_train:n_train+n_cv]
    cv_targets = targets[n_train:n_train+n_cv]

    test_data = data[n_train+n_cv:]
    test_targets = targets[n_train+n_cv:]

    return [train_data[:,:], train_targets[:,:]], [cv_data[:,:], cv_targets[:,:]], [test_data[:,:], test_targets[:,:]]

def prep_data_for_esn(data, targets):
    # adds a bias term to training data
    # squashes the domain of the training data between 0 and 1 (excluding 0 and 1)
    for d in data:
        d.append(1.0) # add bias term
    data = np.asarray(data)
    for i in range( len(data[0]) - 1 ): # do not squash bias
        data[:, i] = squash(data[:,i])

    targets = np.asarray(targets)
    return split_data(data, targets)

def threshold(pred, t_acc, t_brak):
    # a BAD function
    if len(pred.shape)>1:
        for i in range(len(pred[:,0])):
            if pred[i,0] > t_acc:
                pred[i,0] = 1
            else:
                pred[i,0] = 0
        for i in range(len(pred[:,1])):
            if pred[i,1] > t_brak:
                pred[i,1] = 1
            else:
                pred[i,1] = 0
    else:
        if pred[0] > t_acc:
            pred[0] = 1
        else:
            pred[0] = 0
        if pred[1] > t_brak:
            pred[1] = 1
        else:
            pred[1] = 0
    return pred

def correct_dimensions(s, targetlength):
    """checks the dimensionality of some numeric argument s, broadcasts it
       to the specified length if possible.

    Args:
        s: None, scalar or 1D array
        targetlength: expected length of s

    Returns:
        None if s is None, else numpy vector of length targetlength
    """
    if s is not None:
        s = np.array(s)
        if s.ndim == 0:
            s = np.array([s] * targetlength)
        elif s.ndim == 1:
            if not len(s) == targetlength:
                raise ValueError("arg must have length " + str(targetlength))
        else:
            raise ValueError("Invalid argument")
    return s


class ESN():

    def __init__(self, n_inputs, n_outputs, n_reservoir=200,
                 spectral_radius=0.95, sparsity=0, noise=0.001, input_shift=None,
                 input_scaling=None, teacher_forcing=True, feedback_scaling=None,
                 teacher_scaling=None, teacher_shift=None,
                 out_activation=lambda x: x, inverse_out_activation=lambda x: x,
                 random_state=None, silent=True, BAD=False):
        """
        Args:
            n_inputs: nr of input dimensions
            n_outputs: nr of output dimensions
            n_reservoir: nr of reservoir neurons
            spectral_radius: spectral radius of the recurrent weight matrix
            sparsity: proportion of recurrent weights set to zero
            noise: noise added to each neuron (regularization)
            input_shift: scalar or vector of length n_inputs to add to each
                        input dimension before feeding it to the network.
            input_scaling: scalar or vector of length n_inputs to multiply
                        with each input dimension before feeding it to the netw.
            teacher_forcing: if True, feed the target back into output units
            teacher_scaling: factor applied to the target signal
            teacher_shift: additive term applied to the target signal
            out_activation: output activation function (applied to the readout)
            inverse_out_activation: inverse of the output activation function
            random_state: positive integer seed, np.rand.RandomState object,
                          or None to use numpy's builting RandomState.
            silent: supress messages
            BAD: adjustments made by Bram Akash Dmitrii 17-11-2017 in order to experiment with PyESN for TORCS
        """
        # check for proper dimensionality of all arguments and write them down.
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise
        self.input_shift = correct_dimensions(input_shift, n_inputs)
        self.input_scaling = correct_dimensions(input_scaling, n_inputs)

        self.teacher_scaling = teacher_scaling
        self.teacher_shift = teacher_shift

        self.out_activation = out_activation
        self.inverse_out_activation = inverse_out_activation
        self.random_state = random_state
        self.BAD = BAD

        # the given random_state might be either an actual RandomState object,
        # a seed or None (in which case we use numpy's builtin RandomState)
        if isinstance(random_state, np.random.RandomState):
            self.random_state_ = random_state
        elif random_state:
            try:
                self.random_state_ = np.random.RandomState(random_state)
            except TypeError as e:
                raise Exception("Invalid seed: " + str(e))
        else:
            self.random_state_ = np.random.mtrand._rand

        self.teacher_forcing = teacher_forcing
        self.silent = silent
        self.initweights()

        if self.BAD:
            self.t_acc = self.BAD[0]
            self.t_brak = self.BAD[1]

    def initweights(self):
        # initialize recurrent weights:
        # begin with a random matrix centered around zero:
        W = self.random_state_.rand(self.n_reservoir, self.n_reservoir) - 0.5
        # delete the fraction of connections given by (self.sparsity):
        W[self.random_state_.rand(*W.shape) < self.sparsity] = 0
        # compute the spectral radius of these weights:
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        # rescale them to reach the requested spectral radius:
        self.W = W * (self.spectral_radius / radius)

        # random input weights:
        self.W_in = self.random_state_.rand(
            self.n_reservoir, self.n_inputs) * 2 - 1
        # random feedback (teacher forcing) weights:
        self.W_feedb = self.random_state_.rand(
            self.n_reservoir, self.n_outputs) * 2 - 1

    def _update(self, state, input_pattern, output_pattern):
        """performs one update step.

        i.e., computes the next network state by applying the recurrent weights
        to the last state & and feeding in the current input and output patterns
        """
        import numpy as np
        # print(np)
        # print("self.teacher_forcing = ", self.teacher_forcing)

        # print(state.shape)
        # print(input_pattern.shape)
        # print(output_pattern.shape)
        # print("$"*10)


        if self.teacher_forcing:
            # print(np)
            # print("asdfasf")
            preactivation = (np.dot(self.W, state)
                             + np.dot(self.W_in, input_pattern)
                             + np.dot(self.W_feedb, output_pattern))
        else:
            preactivation = (np.dot(self.W, state)
                             + np.dot(self.W_in, input_pattern))

        return (np.tanh(preactivation)
                + self.noise * (self.random_state_.rand(self.n_reservoir) - 0.5))

    def _scale_inputs(self, inputs):
        """for each input dimension j: multiplies by the j'th entry in the
        input_scaling argument, then adds the j'th entry of the input_shift
        argument."""
        if self.input_scaling is not None:
            inputs = np.dot(inputs, np.diag(self.input_scaling))
        if self.input_shift is not None:
            inputs = inputs + self.input_shift
        return inputs

    def _scale_teacher(self, teacher):
        """multiplies the teacher/target signal by the teacher_scaling argument,
        then adds the teacher_shift argument to it."""
        if self.teacher_scaling is not None:
            teacher = teacher * self.teacher_scaling
        if self.teacher_shift is not None:
            teacher = teacher + self.teacher_shift
        return teacher

    def _unscale_teacher(self, teacher_scaled):
        """inverse operation of the _scale_teacher method."""
        if self.teacher_shift is not None:
            teacher_scaled = teacher_scaled - self.teacher_shift
        if self.teacher_scaling is not None:
            teacher_scaled = teacher_scaled / self.teacher_scaling
        return teacher_scaled

    def fit(self, inputs, outputs, inspect=False):
        """
        Collect the network's reaction to training data, train readout weights.

        Args:
            inputs: array of dimensions (N_training_samples x n_inputs)
            outputs: array of dimension (N_training_samples x n_outputs)
            inspect: show a visualisation of the collected reservoir states

        Returns:
            the network's output on the training data, using the trained weights
        """
        # transform any vectors of shape (x,) into vectors of shape (x,1):
        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (len(inputs), -1))
        if outputs.ndim < 2:
            outputs = np.reshape(outputs, (len(outputs), -1))
        # transform input and teacher signal:
        inputs_scaled = self._scale_inputs(inputs)
        teachers_scaled = self._scale_teacher(outputs)

        if not self.silent:
            print("harvesting states...")
        # step the reservoir through the given input,output pairs:
        states = np.zeros((inputs.shape[0], self.n_reservoir))
        for n in range(1, inputs.shape[0]):
            states[n, :] = self._update(states[n - 1], inputs_scaled[n, :],
                                        teachers_scaled[n - 1, :])

        # learn the weights, i.e. find the linear combination of collected
        # network states that is closest to the target output
        if not self.silent:
            print("fitting...")
        # we'll disregard the first few states:
        transient = min(int(inputs.shape[1] / 10), 100)
        # include the raw inputs:
        extended_states = np.hstack((states, inputs_scaled))
        # Solve for W_out:
        self.W_out = np.dot(np.linalg.pinv(extended_states[transient:, :]),
                            self.inverse_out_activation(teachers_scaled[transient:, :])).T

        # remember the last state for later:
        self.laststate = states[-1, :]
        self.lastinput = inputs[-1, :]
        self.lastoutput = teachers_scaled[-1, :]

        # optionally visualize the collected states
        if inspect:
            from matplotlib import pyplot as plt
            # (^-- we depend on matplotlib only if this option is used)
            plt.figure(
                figsize=(states.shape[0] * 0.0025, states.shape[1] * 0.01))
            plt.imshow(extended_states.T, aspect='auto',
                       interpolation='nearest')
            plt.colorbar()

        if not self.silent:
            print("training error:")
        # apply learned weights to the collected states:
        pred_train = self._unscale_teacher(self.out_activation(
            np.dot(extended_states, self.W_out.T)))

        if self.BAD:
            pred_train = threshold(pred_train, self.t_acc, self.t_brak)

        if not self.silent:
            print(np.sqrt(np.mean((pred_train - outputs)**2)))
        return pred_train

    def predict(self, inputs, continuation=True):
        """
        Apply the learned weights to the network's reactions to new input.

        Args:
            inputs: array of dimensions (N_test_samples x n_inputs)
            continuation: if True, start the network from the last training state

        Returns:
            Array of output activations
        """
        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (len(inputs), -1))
        n_samples = inputs.shape[0]

        if continuation:
            laststate = self.laststate
            lastinput = self.lastinput
            lastoutput = self.lastoutput
        else:
            laststate = np.zeros(self.n_reservoir)
            lastinput = np.zeros(self.n_inputs)
            lastoutput = np.zeros(self.n_outputs)

        inputs = np.vstack([lastinput, self._scale_inputs(inputs)])
        states = np.vstack(
            [laststate, np.zeros((n_samples, self.n_reservoir))])
        outputs = np.vstack(
            [lastoutput, np.zeros((n_samples, self.n_outputs))])

        for n in range(n_samples):
            states[
                n + 1, :] = self._update(states[n, :], inputs[n + 1, :], outputs[n, :])
            outputs[n + 1, :] = self.out_activation(np.dot(self.W_out,
                                                           np.concatenate([states[n + 1, :], inputs[n + 1, :]])))
            if self.BAD:
                outputs[n+1,:] = threshold(outputs[n+1,:], self.t_acc, self.t_brak)
                # one BAD rule says that if there is both braking and acceleration, braking has privilege
                if outputs[n+1, 1] == 1:
                    outputs[n+1, 0] = 0

        return self._unscale_teacher(self.out_activation(outputs[1:]))

    def race(self, sens_vec, prev_control, state):
        """
        BAD adjustment; racing for TORCS

        pre-condition; the network has been trained

        the fit() method takes the whole timeseries as input and gives a whole timeseries as output
        we would like to give one timepoint as input and receive an output
        at the next stage, it should still remember previous outputs in determining its new output

        self; a trained ESN
        sens_vec; a 22d vector of sensor data
        prev_control; the previous commands the racer gave as control
        state; the reservoir and its values
        output; a 3d BAD vector of control commands
        """
        import numpy as np
        state = self._update(state, sens_vec, prev_control)
        controls = self.out_activation(np.dot(self.W_out, np.concatenate([state, sens_vec])))

        if self.BAD:
            controls = threshold(controls, self.t_acc, self.t_brak)
            # one BAD rule says that if there is both braking and acceleration, braking has privilege
            if controls[0] == 1:
                controls[1] = 0

        return controls, state


best_param = [.7, .4, .1, .2]

rng = np.random.RandomState(42)
best_t = [.85, .35]

spd_data, spd_targets = preprocess.read_dataset(path="/home/akashrajkn/Documents/github_projects/ruimte-auto/data/f-speedway.csv")
alp_data, alp_targets = preprocess.read_dataset(path="/home/akashrajkn/Documents/github_projects/ruimte-auto/data/alpine-1.csv")
aal_data, aal_targets = preprocess.read_dataset(path="/home/akashrajkn/Documents/github_projects/ruimte-auto/data/aalborg.csv")

spd_train, spd_cv, spd_test = prep_data_for_esn(spd_data, spd_targets)
alp_train, alp_cv, alp_test = prep_data_for_esn(alp_data, alp_targets)
aal_train, aal_cv, aal_test = prep_data_for_esn(aal_data, aal_targets)

spec_rad = best_param[0]
spars = best_param[0]
nos = best_param[0]
tsh = best_param[0]

esn = ESN(n_inputs = 22,
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

all_train_data = list(spd_train[0])
all_train_data.extend(list(aal_train[0]))
all_train_data.extend(list(alp_train[0]))
all_train_data = np.asarray(all_train_data)
all_train_targ = list(spd_train[1])
all_train_targ.extend(list(aal_train[1]))
all_train_targ.extend(list(alp_train[1]))
all_train_targ = np.asarray(all_train_targ)

_ = esn.fit(all_train_data, all_train_targ)

spd_test_sensordata = spd_test[0]
for i in range(len(spd_test_sensordata)):
    sens_vec = spd_test_sensordata[i,:]
    control, state = esn.race(sens_vec, control, state)
    #print(control)

import dill as pickle
# import pickle
def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

# sample usage
save_object(esn, 'esn.pkl')