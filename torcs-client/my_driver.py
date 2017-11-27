import os
import sys
import torch
import pickle
import numpy as np

from torch.autograd import Variable

from pytocl.driver import Driver
from pytocl.car import State, Command

import pytocl.echo_state_network as echo_state_network
from pytocl.neural_net import FeedForwardRegression

sys.modules['echo_state_network'] = echo_state_network


class MyDriver(Driver):
    def __init__(self):
        realpath = os.path.dirname(os.path.realpath(__file__))

        # NN
        self.nn_model = FeedForwardRegression()
        self.nn_model.load_state_dict(torch.load(realpath + '/../models/neural_net_regression.pkl'))

        # ESN
        with open(realpath + '/../models/esn.pkl', 'rb') as f:
            esn_binary = pickle.load(f)

        self.esn =  esn_binary
        self.reservoir = np.zeros(200)
        self.control = np.zeros(3)

    def BAD(self, control, acc = .5, brake = .5, privilege = "brak"):
        '''
        Bram Akash Dmitrii
        threshold acceleration and braking
        acceleration/braking privilege if both are 1
        '''
        if control[0][0] > acc:
            control[0][0] = 1
        else:
            control[0][0] = 0

        if control[0][1] > brake:
            control[0][1] = 1
        else:
            control[0][1] = 0

        if privilege == "brak" and control[0][1] == 1:
            control[0][0] = 0
        if privilege == "acc" and control[0][0] == 1:
            control[0][1] = 0

        return control

    def convert_carstate_to_array(self, carstate, proc='nn'):
        '''
        Convert the carstate to np array
        '''
        speed = carstate.speed_x
        track_position = carstate.distance_from_center
        angle = carstate.angle
        sensors = list(carstate.distances_from_edge)

        # FIXME: this is not a good design
        if proc=='nn':
            carstate_array = np.array([[speed, angle, track_position] + sensors], dtype=np.float32)
        elif proc=='esn':
            carstate_array = np.array([speed, angle, track_position] + sensors + [1.0])  # 1.0 is the bias term

        return carstate_array

    def reservoir_computing(self, carstate):
        '''
        Echo state network
        '''
        sensors = self.convert_carstate_to_array(carstate, proc='esn')
        self.control, self.reservoir = self.esn.race(sensors, self.control, self.reservoir)
        predicted = self.BAD([self.control], acc=.1, brake=.5, privilege="acc")

        return predicted

    def neural_network(self, carstate):
        '''
        Neural network
        '''
        sensors = self.convert_carstate_to_array(carstate)
        predicted = self.nn_model(Variable(torch.from_numpy(sensors))).data.numpy()
        predicted = self.BAD(predicted, acc=.1, brake=.35, privilege="acc")

        return predicted

    def drive(self, carstate: State) -> Command:
        '''
        Custom Drive Function
        '''
        # predicted = self.neural_network(carstate)
        predicted = self.reservoir_computing(carstate)

        command = Command()
        command.accelerator = predicted[0][0]
        command.brake = predicted[0][1]
        command.steering = predicted[0][2]

        # FIXME: Gear changing is not correct

        # if carstate.rpm < 2500:
        #     command.gear = carstate.gear - 1

        if not command.gear:
            command.gear = carstate.gear or 1

        if carstate.rpm > 1500:
            command.gear = carstate.gear + 1

        return command
