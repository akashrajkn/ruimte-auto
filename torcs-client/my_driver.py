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

        self.stuck = False
        self.stuck_counter = 0

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
        # steering -1 is right, steering 1 is left

        # competence levels:
        # 0 move
        # 1 don't drive backwards on the track
        # 2 if off-track, get on track
        # 3 if on-track, finish the lap (by NN or ESN)
        # 4 smarter stuff... (RL?)

        # NN_MODEL
        x_test = self.convert_carstate_to_array(carstate)
        predicted = self.nn_model(Variable(torch.from_numpy(x_test))).data.numpy()
        predicted = self.BAD(predicted, t_acc=.1, t_brak=.35, privilege="acc")

        command = Command()

        command.accelerator = predicted[0][0]
        command.brake = predicted[0][1]
        command.steering = predicted[0][2]

        # if carstate.rpm < 2500:
        #     command.gear = carstate.gear - 1

        # command = Command()
        # command = self.reservoir_computing(carstate, command)

        if not command.gear:
            command.gear = carstate.gear or 1

        # to do: think of a smart way to change gear
        if carstate.rpm > 2500:
            command.gear = 1 + carstate.gear
        # if carstate.rpm > 1500 and carstate.rpm < 3000:
        #     command.gear = 1
        # if carstate.rpm > 3000 and carstate.rpm < 4000:
        #     command.gear = 2
        # if carstate.rpm > 4000 and carstate.rpm < 5000:
        #     command.gear = 3
        # if carstate.rpm > 5000 and carstate.rpm < 6000:
        #     command.gear = 4
        # if carstate.rpm > 6000 and carstate.rpm < 7000:
        #     command.gear = 5
        # if carstate.rpm > 7000:
        #     command.gear = 6

        # determine which competence level is required here
        # low competence levels have higher priority over
        # high competence levels if their criteria are met

        # competence level 2
        # if off-track, get back to the track
        if carstate.distance_from_center > .95:
            command.steering = -.3
        if carstate.distance_from_center < -.95:
            command.steering = .3

        # competence level 1
        # don't drive backwards
        if carstate.angle < -70:
            command.accelerator = .2
            command.brake = 0
            command.gear = 1
            command.steering = -.5
        if carstate.angle > 70:
            command.accelerator = .2
            command.brake = 0
            command.gear = 1
            command.steering = .5

        # competence level 0
        # if stationary; move
        total_speed = np.sqrt(carstate.speed_x**2+carstate.speed_y**2+carstate.speed_z**2)
        if total_speed < 2:
            command.accelerator = .4
            command.brake = 0
            command.gear = 1

        # competence level -1
        # if stuck, get unstuck
        if command.accelerator > 0 and carstate.speed_x < 1:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            self.stuck = False
        if self.stuck_counter > 200:
            self.stuck = True
            command.gear = -1
            command.accelerator = -.4

        print('--------')
        if self.stuck:
            print("STUCK!")
            print(command)

        return command

    def check_for_stuck(self, carstate):
        sensors = self.convert_carstate_to_array(carstate)
        if abs(command.accelerator) > .1 and carstate.speed_x < 1 and sensors[9] < 2.0:
            self.stuck_counter += 1

    def am_i_stuck(self):
        if self.stuck_counter > 100:
            return True
        else:
            return False
