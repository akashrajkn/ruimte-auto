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
        runstats_id = 'runstats-2017-12-01-13-31-44-861522'
        runstats_path = '../src/baselines/runstats/' + runstats_id + '/model_weights.ckpt'
        saver.restore(self.sess, runstats_path)

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

    def drive(self, carstate: State) -> Command:
        '''
        Custom Drive Function
        '''
        predicted = self.ddpg_driver(carstate)

        command = Command()
        # Order of controls for DDPG (note: different than for FNN and ESN):
        command.steering = predicted[0][0]
        command.accelerator = predicted[0][1]
        command.brake = predicted[0][2]

        # Automatic Transmission like in snakeoil
        command.gear=1
        if carstate.speed_x>50:
            command.gear=2
        if carstate.speed_x>80:
            command.gear=3
        if carstate.speed_x>110:
            command.gear=4
        if carstate.speed_x>140:
            command.gear=5
        if carstate.speed_x>170:
            command.gear=6

        return command
