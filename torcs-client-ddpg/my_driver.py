import os
import sys
import json
import math
import numpy as np
import collections as col
from pytocl.driver import Driver
from pytocl.car import State, Command, DEGREE_PER_RADIANS, MPS_PER_KMH

import tensorflow as tf
from pytocl.ddpg.models import Actor, Critic
from pytocl.ddpg.memory import Memory
from pytocl.ddpg.ddpg import DDPG

class MyDriver(Driver):
    def __init__(self):
        realpath = os.path.dirname(os.path.realpath(__file__))

        self.control = np.zeros(3)
        self.car_number = None
        self.friend = None
        self.steps = 0

        # bram variables for rule-based low competence levels:
        self.bully = False
        self.speed_is_standstill = 14 # under which speed we consider ourselfs to be 'standing still'
        self.standstill_counter = 0
        self.angle_limit = 26
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

        self.champion_agent = DDPG(actor, critic, memory, (29,),  (3,),
                    gamma=0.99, tau=0.01, normalize_returns=False,
                    normalize_observations=True,
                    batch_size=64, action_noise=action_noise, param_noise=param_noise,
                    critic_l2_reg=1e-2, actor_lr=1e-4, critic_lr=1e-3,
                    enable_popart=False, clip_norm=None, reward_scale=1.)

        saver = tf.train.Saver()
        self.sess = tf.Session()

        # Restore variables from disk.
        runstats_id_champ = 'Dimas_champion_trained_11hrs'
        runstats_path_champ = '../src/baselines/runstats/' + runstats_id_champ + '/model_weights.ckpt'
        saver.restore(self.sess, runstats_path_champ)

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
        command = self.champion_agent.act(obs, self.sess)
        # All actions are predicted in [-1, 1]; normalizing back:
        command[1] = (command[1]+1)/2
        command[2] = (command[2]+1)/2
        #command[2] = 0
        command = [command]
        return command

    def repel(self, carstate):
        pass

    def attract(self, carstate):
        pass

    def is_bully(self, carstate, friend_carstate):
        '''
        For swarm intelligence, check if the current car is bully or champion
        '''

        if carstate is None:
            return False

        if friend_carstate is None or friend_carstate.get('distance_from_start') is None:
            return False

        # If the car is in the end, it is a champion
        if int(carstate.sensor_dict['racePos']) > 7:
            return False

        # If distance between cars is more than 60m, set the car as bully
        distance_between_cars = int(friend_carstate['distance_from_start']) - carstate.distance_from_start

        if distance_between_cars > 60:
            return True
        if distance_between_cars < -60:
            return False

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

    def comp_4_ddpg(self, carstate, show):
        # DDPG_model
        control_vec = np.full(4, None)
        predicted = self.ddpg_driver(carstate)

        # Order of controls for DDPG (note: different than for FNN and ESN):
        control_vec[2] = predicted[0][0] # steering
        control_vec[0] = predicted[0][1] # accelerator
        control_vec[1] = predicted[0][2] # brake
        if show:
            print("Competence level 4 engaged. Ddpg at the wheel")

        return control_vec

    def rule_based_bully(self, carstate, control_vec, show):
        # if carstate.opponents[0] < 20 or carstate.opponents[35] < 20:
        #     control_vec[0] *= .8
        #     if show:
        #         print("they're behind me, I'll slow down")
        #
        # if any(carstate.opponents[1:4] < 12):
        #     control_vec[0] *= .9
        #     control_vec[2] -= .05
        # if any(carstate.opponents[30:34] < 12):
        #     control_vec[0] *= .9
        #     control_vec[2] += .05
        #
        # if any(carstate.opponents[5:10] < 4):
        #     if carstate.opponents[10] < 3:
        #         control_vec[0] = 1
        #     control_vec[2] -= .12
        #
        # if any(carstate.opponents[24:29] < 4):
        #     if carstate.opponents[24] < 3:
        #         control_vec[0] = 1
        #     control_vec[2] += .12
        #
        if show:
            print("I am the bully")

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

    def swarm_communication(self, carstate):
        '''
        For swarm intelligence - Establishing communication
        1. Checks if car_number is assignend. If not assign current racePosition
        2. If friend number is unknown
            - If BAD file has both entries, then reads and assigns friend number
        3. For each step (10m), car updates its racePos and distance_from_start
        '''
        # Happens only one time
        if self.car_number is None:
            self.car_number = int(carstate.sensor_dict['racePos'])
            with open('BAD', 'a') as f:
                f.write(str(self.car_number)+ ' ')

        # Happens only one time
        if self.car_number is not None and self.friend is None:
            with open('BAD', 'r') as f:
                text = f.read()
            cars = text.strip().split()
            if len(cars) == 2:
                self.friend = cars[0] if cars[1] == str(self.car_number) else cars[1]
                self.friend = int(self.friend)

        # Update information to file only in certain steps
        next_step = math.floor(carstate.distance_from_start / 10)

        if self.car_number and self.friend:
            if self.steps != next_step:
                self.steps = next_step
                information = {
                    'racePos': carstate.sensor_dict['racePos'],
                    'distance_from_start': carstate.distance_from_start
                }
                filepath = 'BAD1.json'

                if self.car_number < self.friend:
                    filepath = 'BAD2.json'
                with open(filepath, 'w+') as f:
                    json.dump(information, f)

    def drive(self, carstate: State) -> Command:
        #'''
        #Custom Drive Function
        #'''

        #self.swarm_communication(carstate)

        command = Command()

        if abs(carstate.speed_x) < self.speed_is_standstill:
            self.standstill_counter += 1
        else:
            self.standstill_counter = 0

        friend_carstate = None
        # Read friend carstate
        if self.car_number and self.friend:
            if self.car_number > self.friend:
                filepath = 'BAD2.json'
            else:
                filepath = 'BAD1.json'
            try:
                with open(filepath, 'r') as f:
                    friend_carstate = json.load(f)
            except:
                pass

        self.bully = self.is_bully(carstate, friend_carstate)

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
                            if self.bully:
                                control_vec = self.rule_based_bully(carstate, control_vec, show)
                            if any(control_vec == None):
                                inferior = self.gearbox(carstate)
                                control_vec = self.cva_priority(control_vec, inferior)

        #control_vec = self.privilege(control_vec)

        command.accelerator = control_vec[0]
        command.brake = control_vec[1]
        command.steering = control_vec[2]
        command.gear = control_vec[3]

        return command
