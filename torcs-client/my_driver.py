import os
import torch
import numpy as np
import dill as pickle

from torch.autograd import Variable

from pytocl.driver import Driver
from pytocl.car import State, Command

from pytocl.neural_net import FeedForwardRegression
# from pytocl.echo_state_network import ESN


class MyDriver(Driver):
    def __init__(self):
        realpath = os.path.dirname(os.path.realpath(__file__))

        self.nn_model = FeedForwardRegression()
        self.nn_model.load_state_dict(torch.load(realpath + '/../models/neural_net_regression.pkl'))

        # # ESN
        # with open('realpath + '/../models/esn.pkl', 'rb') as f:
        #     esn_binary = pickle.load(f)
        #
        # self.esn =  esn_binary
        # self.reservoir = np.zeros(200)
        # self.control = np.zeros(3)

    def reservoir_computing(self, carstate, command):

        sensors = self.convert_carstate_to_array(carstate)
        self.control, self.reservoir = self.esn.race(sensors, self.control, self.reservoir)

        command.accelerator = self.control[0]
        command.brake = self.control[1]
        command.steering = self.control[2]

        return command

    def convert_carstate_to_array(self, carstate):
        '''
        Convert the carstate to np array
        '''
        speed = carstate.speed_x
        track_position = carstate.distance_from_center
        angle = carstate.angle

        sensors = list(carstate.distances_from_edge)
        # FIXME: NN model does not use one of the sensors
        sensors = sensors[:-1]
        carstate_array = np.array([[speed, angle, track_position] + sensors], dtype=np.float32)

        # carstate_array = np.array([speed, angle, track_position] + sensors)

        return carstate_array

    def BAD(self, control_vec, t_acc = .5, t_brak = .5, privilege = "brak"):
        # Bram Akash Dmitrii
        # threshold acceleration and braking
        # acceleration/braking privilege if both are 1

        if control_vec[0][0] > t_acc:
            control_vec[0][0] = 1
        else:
            control_vec[0][0] = 0
        if control_vec[0][1] > t_brak:
            control_vec[0][1] = 1
        else:
            control_vec[0][1] = 0
        if privilege == "brak" and control_vec[0][1] == 1:
            control_vec[0][0] = 0
        if privilege == "acc" and control_vec[0][0] == 1:
            control_vec[0][1] = 0
        return control_vec

    def drive(self, carstate: State) -> Command:
        # NN_MODEL
        x_test = self.convert_carstate_to_array(carstate)
        predicted = self.nn_model(Variable(torch.from_numpy(x_test))).data.numpy()
        predicted = self.BAD(predicted, t_acc=.1, t_brak=.35, privilege="acc")

        command = Command()

        print("-------")
        print(predicted[0])

        command.accelerator = predicted[0][0]
        command.brake = predicted[0][1]
        command.steering = predicted[0][2]

        # if carstate.rpm < 2500:
        #     command.gear = carstate.gear - 1

        # command = Command()
        # command = self.reservoir_computing(carstate, command)

        if not command.gear:
            command.gear = carstate.gear or 1

        if carstate.rpm > 1500:
            command.gear = carstate.gear + 1

        return command
