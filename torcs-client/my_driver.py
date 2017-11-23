import torch
import numpy as np
import dill as pickle

from torch.autograd import Variable

from pytocl.nn_linear_regression import LinearRegression
from pytocl.pyESN_for_TORCS import ESN
from pytocl.driver import Driver
from pytocl.car import State, Command


class MyDriver(Driver):
    def __init__(self):
        input_size = 21
        hidden_size = 40
        output_size = 3
        self.nn_model = LinearRegression(input_size, output_size, hidden_size)
        # self.nn_model.load_state_dict(torch.load('../../models/model_nn_regression.pkl'))

        # ESN
        with open('/home/akashrajkn/Documents/github_projects/ruimte-auto/torcs-client/pytocl/esn.pkl', 'rb') as in_strm:
            datastruct = pickle.load(in_strm)

        self.esn = datastruct
        self.reservoir = np.zeros(200)
        self.control = np.zeros(3)

    def reservoir_computing(self, carstate, command):

        sensors = self.convert_carstate_to_array(carstate)
        self.control, self.reservoir = self.esn.race(sensors, self.control, self.reservoir)

        command.accelerator = self.control[0]
        command.brake = self.control[1]
        command.steering = self.control[2]

        return command

    def convert_carstate_to_array(self, carstate):
        '''
        Convert the carstate to numpy array
        '''
        speed = carstate.speed_x
        track_position = carstate.distance_from_center
        angle = carstate.angle

        sensors = list(carstate.distances_from_edge)
        # FIXME: NN model does not use one of the sensors
        # sensors = sensors[:-1]

        carstate_array = np.array([[speed, angle, track_position] + sensors], dtype=np.float32)

        return carstate_array

    def drive(self, carstate: State) -> Command:
        # # NN_MODEL
        # x_test = self.convert_carstate_to_array(carstate)
        # predicted = self.nn_model(Variable(torch.from_numpy(x_test))).data.numpy()
        #
        # command = Command()
        #
        # command.accelerator = predicted[0][0]
        # command.brake = predicted[0][1]
        # command.steering = predicted[0][2]
        #
        # # if carstate.rpm < 2500:
        # #     command.gear = carstate.gear - 1

        command = Command()
        command = self.reservoir_computing(carstate, command)

        if not command.gear:
            command.gear = carstate.gear or 1

        if carstate.rpm > 1500:
            command.gear = carstate.gear + 1

        return command
