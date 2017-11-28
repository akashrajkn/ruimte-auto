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

        # bram variables:
        self.standstill_counter = 0
        self.engage_comp0 = 36 # after how many epochs standstill comp 0 engages
        self.engage_comp_neg1 = 210 # after how many epochs driving but not moving do we realise we are hitting the wall
        self.dismiss_comp_neg1 = 600 # after how many epochs driving back are we going to go forward again (may be sooner if we hit enough speed)

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

    def comp_minus1(self, carstate, command): # back up
    # bram's implementation
        control_vec = np.full(4, None)
        if self.standstill_counter > self.engage_comp_neg1:
            control_vec = []
            print("competence level -1 engaged. Back up")
            control_vec[3] = -1 # gear
            control_vec[0] = .5 # acc
            control_vec[1] = 0 # brake
            if carstate.distance_from_center > 0:
                control_vec[2] = .5 # steer
            if carstate.distance_from_center < 0:
                control_vec[2] = -.5
        if self.standstill_counter > self.dismiss_comp_neg1:
            print("ive backup up long enough, hopefully there is space in front of me now")
            self.standstill_counter = 4
        return control_vec

    def comp_0(self, carstate, command): # move
    # bram's implementation
        control_vec = np.full(4, None)
        if abs(carstate.speed_x) < 5:
            self.standstill_counter += 1
        else:
            self.standstill_counter = 0
        if self.standstill_counter > self.engage_comp0 and self.standstill_counter < self.engage_comp_neg1:
            print("competence level 0 engaged. Move, dammit.")
            control_vec[3] = 1 # gear
            control_vec[0] = .5 # acc
            control_vec[1] = 0 # brak
            if carstate.distance_from_center > 0.1:
                control_vec[2] = -.1 # steer
            if carstate.distance_from_center < -0.1:
                control_vec[2] = .1 # steer
        return control_vec

    def comp_1(self, carstate, command): # face the right way, don't drive backwards
    # bram's implementation
        control_vec = np.full(4, None)
        if carstate.angle < -70 and carstate.angle > 70:
            if carstate.speed_x < 30:
                control_vec[3] = 1 # gear
                control_vec[0] = .2 # acc
                control_vec[1] = 0 # brak
            else:
                control_vec[0] = 0 # acc
                control_vec[1] = .5 # brak
        if carstate.angle < -70:
            control_vec[2] = -.5 # steer
        if carstate.angle > 70:
            control_vec[2] = .5 # steer
        return control_vec

    def comp_2(self, carstate, command): # if off-track, steer to the track
    # bram's implementation
        control_vec = np.full(4, None)
        if carstate.distance_from_center > 1:
            control_vec[2] = -.2 # steer
        if carstate.distance_from_center < -1:
            control_vec[2] = .2 # steer
        return control_vec

    def comp_3_fNN(self, carstate):
        # NN_MODEL
        control_vec = np.full(4, None)
        x_test = self.convert_carstate_to_array(carstate)
        predicted = self.nn_model(Variable(torch.from_numpy(x_test))).data.numpy()
        #predicted = self.BAD(predicted, t_acc=.1, t_brak=.35, privilege="acc")
        predicted = self.privilege(predicted)
        control_vec[3] = carstate.gear or 1 # gear
        control_vec[:3] = predicted[0]
        return control_vec

    def privilege(self, control):
        ratio = 1.8 # we find acceleration "ratio" times more important than brake
        if control[0][0]*ratio > control[0][1]:
            control[0][1] = 0
        if control[0][1] > control[0][0]*ratio:
            control[0][0] = 0
        return control

    def cva_priority(self, superior, inferior): # control vector according to priority
        # superior [ 1, 2, None, 4 ]
        # inferior [ 5, None, 7, 8 ]
        # return [1, 2, 7, 4]
        for idx, value in enumerate(superior): # try this in list comprehension, it will be 1 line instead of 3
            if value == None:
                superior[idx] = inferior[idx]
        return superior

    def drive(self, carstate: State) -> Command:
        # competence levels:
        # 0 move
        # 1 don't drive backwards on the track
        # 2 stay on the track
        # 3 if on-track, finish the lap (by NN or ESN)
        # 4 smarter stuff... (RL?)

        command = Command()

        if not command.gear:
            command.gear = carstate.gear or 1

        # to do: think of a smart way to change gear
        if carstate.rpm > 2500:
            command.gear = 1 + carstate.gear

        # determine which competence level is required here
        # low competence levels have higher priority over
        # high competence levels if their criteria are met

        # TO DO; properly suppres low priority and determine which controls
        # overwriting command.something gives bad things
        control_vec = self.comp_minus1(carstate, command)
        if any(control_vec == None):
            print('-----------')
            print("control_vec", control_vec)
            inferior = self.comp_0(carstate, command)
            control_vec = self.cva_priority(control_vec, inferior)
            if any(control_vec == None):
                inferior = self.comp_1(carstate, command)
                control_vec = self.cva_priority(control_vec, inferior)
                if any(control_vec == None):
                    inferior = self.comp_2(carstate, command)
                    control_vec = self.cva_priority(control_vec, inferior)
                    if any(control_vec == None):
                        inferior = self.comp_3_fNN(carstate)
                        control_vec = self.cva_priority(control_vec, inferior)

        command.accelerator = control_vec[0]
        command.brake = control_vec[1]
        command.steering = control_vec[2]
        command.gear = control_vec[3]

        # # competence level 2
        # # if off-track, get back to the track
        # self.comp_2(carstate, command)
        #
        # # competence level 1
        # # face the right way
        # self.comp_1(carstate, command)
        #
        # # competence level 0
        # # move
        # self.comp_0(carstate, command)
        #
        # # competence level -1
        # # back up if we are facing a wall
        # self.comp_minus1(carstate, command)

        print('-----------------')
        print("gear", carstate.gear)
        print("speed", carstate.speed_x)
        print("acc", command.accelerator)
        print("standstillcounter", self.standstill_counter)

        return command
