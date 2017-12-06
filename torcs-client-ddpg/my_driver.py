import os
import sys
import pickle
import numpy as np
import collections as col
from pytocl.driver import Driver
from pytocl.car import State, Command

import tensorflow as tf
from pytocl.ddpg.models import Actor, Critic
from pytocl.ddpg.memory import Memory
from pytocl.ddpg.ddpg import DDPG


class MyDriver(Driver):
    def __init__(self):
        realpath = os.path.dirname(os.path.realpath(__file__))

        self.control = np.zeros(3)

        # bram variables for rule-based low competence levels:
        self.speed_is_standstill = 13 # under which speed we consider ourselfs to be 'standing still'
        self.standstill_counter = 0
        self.engage_comp0 = 36 # after how many epochs standstill comp 0 engages
        self.engage_comp_neg1 = 180 # after how many epochs driving but not moving do we realise we are hitting the wall
        self.dismiss_comp_neg1 = 580 # after how many epochs driving back are we going to go forward again (may be sooner if we hit enough speed)

        # DDPG
        nb_actions = 3
        action_noise = None
        param_noise = None
        memory = Memory(limit=int(1e6), action_shape=(3,), observation_shape=(29,))
        critic = Critic(layer_norm=True)
        actor = Actor(nb_actions, layer_norm=True)
        self.agent = DDPG(actor, critic, memory, (29,),  (3,),
                    gamma=0.99, tau=0.01, normalize_returns=False,
                    normalize_observations=True,
                    batch_size=64, action_noise=action_noise, param_noise=param_noise,
                    critic_l2_reg=1e-2, actor_lr=1e-4, critic_lr=1e-3,
                    enable_popart=False, clip_norm=None, reward_scale=1.)

        saver = tf.train.Saver()
        self.sess = tf.Session()
        # Restore variables from disk.
        runstats_id = 'runstats-2017-12-01-13-31-44-861522' # Dima's first thing from 1dec
        #runstats_id = 'first_10_min_of_bully_training_from_scratch'
        #runstats_id = 'first_7hrs_of_bully_training'
        runstats_path = '../src/baselines/runstats/' + runstats_id + '/model_weights.ckpt'
        saver.restore(self.sess, runstats_path)

        # For Swarm intelligence
        communication_file = '/tmp/BAD'

        self.bully = False

        for i in range(1, 10):
            try:
                self.car_number = i
                os.mkfifo(communication_file + str(i))
                break
            except OSError as e:
                print("OS Error: ", e)

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
        if proc=='ddpg':
            obs = self.make_observation(carstate.sensor_dict)
            carstate_array = np.hstack((obs.angle,
                                        obs.track,
                                        obs.trackPos,
                                        obs.speedX,
                                        obs.speedY,
                                        obs.speedZ,
                                        obs.wheelSpinVel/100.0,
                                        obs.rpm))
        return carstate_array

    def make_observation(self, raw_obs):
        names = ['focus',  'speedX', 'speedY', 'speedZ', 'angle', 'damage',
                 'opponents',
                 'rpm',
                 'track',
                 'trackPos',
                 'wheelSpinVel']
        Observation = col.namedtuple('Observaion', names)
        return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                           speedX=np.array(raw_obs['speedX'], dtype=np.float32)/300.0,
                           speedY=np.array(raw_obs['speedY'], dtype=np.float32)/300.0,
                           speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/300.0,
                           angle=np.array(raw_obs['angle'], dtype=np.float32)/3.1416,
                           damage=np.array(raw_obs['damage'], dtype=np.float32),
                           opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                           rpm=np.array(raw_obs['rpm'], dtype=np.float32)/10000,
                           track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                           trackPos=np.array(raw_obs['trackPos'], dtype=np.float32)/1.,
                           wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32))

    def ddpg_driver(self, carstate):
        obs = self.convert_carstate_to_array(carstate, proc='ddpg')
        command = self.agent.act(obs, self.sess)
        # All actions are predicted in [-1, 1]; normalizing back:
        command[1] = (command[1]+1)/2
        command[2] = (command[2]+1)/2
        #command[2] = 0
        command = [command]
        return command

    def comp_minus1(self, carstate, show): # back up
    # bram's implementation
        control_vec = np.full(4, None)
        if self.standstill_counter > self.engage_comp_neg1:
            if show:
                print("competence level -1 engaged. Back up")
            control_vec[3] = -1 # gear
            control_vec[0] = .16 # acc
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
        if carstate.angle < -45 or carstate.angle > 45:
            if carstate.speed_x > 0 and carstate.speed_x < 30:
                control_vec[3] = 1 # gear
                control_vec[0] = .04 # acc
                control_vec[1] = 0 # brak
            else:
                control_vec[3] = 1
                control_vec[0] = 0 # acc
                control_vec[1] = .5 # brak
        print("angle:",carstate.angle)
        if carstate.angle < -70:
            control_vec[2] = -.34 # steer
            if show:
                print("adjust steering")
        if carstate.angle > 70:
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

    def comp_4_ddpg(self, carstate, show):
        # DDPG_model
        control_vec = np.full(4, None)

        predicted = self.ddpg_driver(carstate)

        # Order of controls for DDPG (note: different than for FNN and ESN):
        control_vec[2] = predicted[0][0] # steering
        control_vec[0] = predicted[0][1] # accelerator
        control_vec[1] = predicted[0][2] # brake

        if show and any(control_vec != None):
            print("competence level 4 engaged. DDPG has the steering wheel")

        return control_vec

    def gearbox(self, carstate):
        control_vec = np.full(4, None)
        control_vec[3] = carstate.gear or 1
        if (carstate.rpm > 0 and carstate.gear == 0) or (carstate.rpm < 1000 and carstate.gear == 2):
            control_vec[3] = 1
        if carstate.rpm > 1800 and carstate.gear == 1 or (carstate.rpm < 1000 and carstate.gear == 3):
            control_vec[3] = 2
        if carstate.rpm > 2500 and carstate.gear == 2 or (carstate.rpm < 1500 and carstate.gear == 4):
            control_vec[3] = 3
        if carstate.rpm > 3200 and carstate.gear == 3 or (carstate.rpm < 1800 and carstate.gear == 5):
            control_vec[3] = 4
        if carstate.rpm > 4000 and carstate.gear == 4 or (carstate.rpm < 2000 and carstate.gear == 6):
            control_vec[3] = 5
        if carstate.rpm > 5000 and carstate.gear == 5:
            control_vec[3] = 6
        if carstate.speed_x < -10:
            control_vec[3] = 0
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
        '''
        Custom Drive Function
        '''
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
                            inferior = self.comp_4_ddpg(carstate, show)
                            control_vec = self.cva_priority(control_vec, inferior)
                            if any(control_vec == None):
                                inferior = self.gearbox(carstate)
                                control_vec = self.cva_priority(control_vec, inferior)

        control_vec = self.privilege(control_vec)
        print(control_vec)

        command.accelerator = control_vec[0]
        command.brake = control_vec[1]
        command.steering = control_vec[2]
        command.gear = control_vec[3]

        # # Automatic Transmission like in snakeoil
        # command.gear=1
        # if carstate.speed_x>50:
        #     command.gear=2
        # if carstate.speed_x>80:
        #     command.gear=3
        # if carstate.speed_x>110:
        #     command.gear=4
        # if carstate.speed_x>140:
        #     command.gear=5
        # if carstate.speed_x>170:
        #     command.gear=6

        return command

    def is_friend(self, carstate, friend_carstate):
        '''
        Find if the car is not opponent
        '''
        # TODO: this function is not complete

        if abs(carstate.racePos - friend_carstate.racePos) == 1:
            return True

    def is_bully(self, carstate, friend_carstate):
        '''
        For swarm intelligence, check if the current car is bully or champion
        '''
        if friend_carstate is None:
            return False

        # If the car is in the end, it is a champion
        if carstate.racePos > 8:
            return False

        # If distance between cars is more than 60m, set the car as champion
        distance_between_cars = friend_carstate.distFromStart - carstate.distFromStart
        if (distance_between_cars > 60) or (distance_between_cars < -60):
            return False
