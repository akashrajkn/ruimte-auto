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

        # bram variables for rule-based low competence levels:
        self.speed_is_standstill = 12 # under which speed we consider ourselfs to be 'standing still'
        self.standstill_counter = 0
        self.angle_limit = 22
        self.engage_comp0 = 56 # after how many epochs standstill comp 0 engages
        self.engage_comp_neg1 = 300 # after how many epochs driving but not moving do we realise we are hitting the wall
        self.dismiss_comp_neg1 = 570 # after how many epochs driving back are we going to go forward again (may be sooner if we hit enough speed)

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

    def comp_minus1(self, carstate, show): # back up
    # bram's implementation
        control_vec = np.full(4, None)
        if self.standstill_counter > self.engage_comp_neg1:
            if show:
                print("competence level -1 engaged. Back up")
            control_vec[3] = -1 # gear
            control_vec[0] = .17 # acc
            control_vec[1] = 0 # brake
            if carstate.distance_from_center > 0:
                control_vec[2] = .43 # steer
            if carstate.distance_from_center < 0:
                control_vec[2] = -.43
            #if abs(carstate.angle) > 110:
                #control_vec[2] = -control_vec[2]
        if self.standstill_counter > self.dismiss_comp_neg1:
            if show:
                print("ive backed up up long enough, hopefully there is space in front of me now")
            self.standstill_counter = 0
        return control_vec

    def comp_0(self, carstate, show): # move
        # bram's implementation
        control_vec = np.full(4, None)
        if self.standstill_counter > self.engage_comp0 and self.standstill_counter < self.engage_comp_neg1:
            if show:
                print("competence level 0 engaged. Move, dammit.")
            control_vec[3] = 1 # gear
            control_vec[0] = .5 # acc
            control_vec[1] = 0 # brak
        return control_vec

    def comp_1(self, carstate, show): # face the right way, don't drive backwards
    # bram's implementation
        control_vec = np.full(4, None)
        if abs(carstate.angle)> self.angle_limit:
            if carstate.speed_x > 0 and carstate.speed_x < 30:
                control_vec[3] = 1 # gear
                control_vec[0] = .04 # acc
                control_vec[1] = 0 # brak
            else:
                control_vec[3] = 1
                control_vec[0] = 0 # acc
                control_vec[1] = .5 # brak
        if carstate.angle < -self.angle_limit:
            control_vec[2] = -.34 # steer
            if show:
                print("adjust steering")
        if carstate.angle > self.angle_limit:
            control_vec[2] = .34 # steer
            if show:
                print("adjust steering")
        if show and any(control_vec != None):
            print("competence level 1 engaged. Trying to face the right way")
        return control_vec

    def comp_2(self, carstate, show): # if off-track, steer to the track
    # bram's implementation
        control_vec = np.full(4, None)
        if carstate.distance_from_center > 1.03:
            control_vec[2] = -.4 # steer
        if carstate.distance_from_center < -1.03:
            control_vec[2] = .4 # steer
        if show and any(control_vec != None):
            print("competence level 2 engaged. override steering in order to stay on track or get on track")
        return control_vec

    def comp_3_go(self, carstate, show):
        # if there is nothing ahead, put the pedal to the metal
        control_vec = np.full(4, None)
        if carstate.distances_from_edge[9] > 150 and abs(carstate.angle) < 45 :
            control_vec[0] = 1
        if show and any(control_vec != None):
            print("competence level 3 engaged. pedal to the metal \m/")
        return control_vec

    def reservoir_computing(self, carstate):
        '''
        Echo state network
        '''
        control_vec = np.full(4, None)
        sensors = self.convert_carstate_to_array(carstate, proc='esn')
        self.control, self.reservoir = self.esn.race(sensors, self.control, self.reservoir)
        #predicted = self.BAD([self.control], acc=.1, brake=.5, privilege="acc")
        control_vec[:3] = self.control
        return control_vec

    def neural_network(self, carstate):
        '''
        Neural network
        '''
        control_vec = np.full(4, None)
        sensors = self.convert_carstate_to_array(carstate)
        predicted = self.nn_model(Variable(torch.from_numpy(sensors))).data.numpy()
        #predicted = self.BAD(predicted, acc=.1, brake=.35, privilege="acc")
        control_vec[:3] = predicted
        return control_vec

    def gearbox(self, carstate):
        control_vec = np.full(4, None)
        control_vec[3] = carstate.gear or 1
        if (carstate.rpm > 0 and carstate.gear == 0) or (carstate.rpm < 2000 and carstate.gear == 2):
            control_vec[3] = 1
        if carstate.rpm > 3200 and carstate.gear == 1 or (carstate.rpm < 3000 and carstate.gear == 3):
            control_vec[3] = 2
        if carstate.rpm > 5000 and carstate.gear == 2 or (carstate.rpm < 4000 and carstate.gear == 4):
            control_vec[3] = 3
        if carstate.rpm > 6000 and carstate.gear == 3 or (carstate.rpm < 5000 and carstate.gear == 5):
            control_vec[3] = 4
        if carstate.rpm > 7000 and carstate.gear == 4 or (carstate.rpm < 6000 and carstate.gear == 6):
            control_vec[3] = 5
        if carstate.rpm > 8000 and carstate.gear == 5:
            control_vec[3] = 6
        if carstate.speed_x < -10:
            control_vec[3] = 0

        # # Automatic Transmission like in snakeoil
        # control_vec[3] = 1
        # if carstate.speed_x > 50:
        #     control_vec[3] = 2
        # if carstate.speed_x > 80:
        #     control_vec[3] = 3
        # if carstate.speed_x > 110:
        #     control_vec[3] = 4
        # if carstate.speed_x > 140:
        #     control_vec[3] = 5
        # if carstate.speed_x > 170:
        #     control_vec[3] = 6
        # if carstate.speed_x < -10:
        #     control_vec[3] = 0

        return control_vec

    def privilege(self, control):
        ratio = 1.9 # we find acceleration "ratio" times more important than brake
        if control[0]*ratio > control[1]:
            control[1] = 0
        if control[1] > control[0]*ratio:
            control[0] = 0
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
        #'''
        #Custom Drive Function
        #'''

        command = Command()

        if abs(carstate.speed_x) < self.speed_is_standstill:
            self.standstill_counter += 1
        else:
            self.standstill_counter = 0

        # determine which competence level is required here
        # low competence levels have higher priority over
        # high competence levels if their criteria are met
        show = True
        control_vec = self.comp_minus1(carstate, show)
        if any(control_vec == None):
            inferior = self.comp_0(carstate, show)
            control_vec = self.cva_priority(control_vec, inferior)
            if any(control_vec == None):
                inferior = self.comp_1(carstate, show)
                control_vec = self.cva_priority(control_vec, inferior)
                if any(control_vec == None):
                    inferior = self.comp_2(carstate, show)
                    control_vec = self.cva_priority(control_vec, inferior)
                    if any(control_vec == None):
                        inferior = self.comp_3_go(carstate, show)
                        control_vec = self.cva_priority(control_vec, inferior)
                        if any(control_vec == None):
                            inferior = self.reservoir_computing(carstate)
                            control_vec = self.cva_priority(control_vec, inferior)
                            if show:
                                print("AI has the wheel")
                            if any(control_vec == None):
                                inferior = self.gearbox(carstate)
                                control_vec = self.cva_priority(control_vec, inferior)

        control_vec = self.privilege(control_vec)

        command.accelerator = control_vec[0]
        command.brake = control_vec[1]
        command.steering = control_vec[2]
        command.gear = control_vec[3]

        return command
