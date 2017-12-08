import os
import sys
import json
import math
import numpy as np
import collections as col
from pytocl.driver import Driver
from pytocl.car import State, Command, DEGREE_PER_RADIANS, MPS_PER_KMH

import tensorflow as tf
from pytocl.ddpg.models import Actor_Bully, Critic_Bully, Actor_Champion, Critic_Champion
from pytocl.ddpg.memory import Memory
from pytocl.ddpg.ddpg import DDPG

class MyDriver(Driver):
    def __init__(self):
        realpath = os.path.dirname(os.path.realpath(__file__))
        # DDPG
        nb_actions = 3
        action_noise = None
        param_noise = None
        memory_29 = Memory(limit=int(1e6), action_shape=(3,), observation_shape=(29,))
        memory_65 = Memory(limit=int(1e6), action_shape=(3,), observation_shape=(65,))

        critic_bully = Critic_Bully(layer_norm=True)
        actor_bully = Actor_Bully(nb_actions, layer_norm=True)
        critic_champion = Critic_Champion(layer_norm=True)
        actor_champion = Actor_Champion(nb_actions, layer_norm=True)

        champ_graph = tf.Graph()
        with champ_graph.as_default():
            self.champ_agent = DDPG(actor_champion, critic_champion, memory_29, (29,),  (3,),
                    gamma=0.99, tau=0.01, normalize_returns=False,
                    normalize_observations=True,
                    batch_size=64, action_noise=action_noise, param_noise=param_noise,
                    critic_l2_reg=1e-2, actor_lr=1e-4, critic_lr=1e-3,
                    enable_popart=False, clip_norm=None, reward_scale=1.)

        bull_graph = tf.Graph()
        with bull_graph.as_default():
            self.bull_agent = DDPG(actor_bully, critic_bully, memory_65, (65,),  (3,),
                    gamma=0.99, tau=0.01, normalize_returns=False,
                    normalize_observations=True,
                    batch_size=64, action_noise=action_noise, param_noise=param_noise,
                    critic_l2_reg=1e-2, actor_lr=1e-4, critic_lr=1e-3,
                    enable_popart=False, clip_norm=None, reward_scale=1.)

        self.champ_sess = tf.Session(graph=champ_graph)
        self.bull_sess = tf.Session(graph=bull_graph)

        with self.champ_sess.as_default():
            with champ_graph.as_default():
                tf.global_variables_initializer().run()
                champ_saver = tf.train.Saver(tf.global_variables())
                # champ_ckpt = tf.train.get_checkpoint_state(args.save_dir)

                runstats_id_champ = 'Dimas_champion_trained_11hrs'
                runstats_path_champ = '../src/baselines/runstats/' + runstats_id_champ + '/model_weights.ckpt'
                #saver.restore(self.sess, runstats_path_champ)

                #champ_saver.restore(champ_sess, model_ckpt.model_checkpoint_path)
                champ_saver.restore(self.champ_sess, runstats_path_champ)

        with self.bull_sess.as_default():
            with bull_graph.as_default():
                tf.global_variables_initializer().run()
                bull_saver = tf.train.Saver(tf.global_variables())
                # adv_ckpt = tf.train.get_checkpoint_state(adv_args.save_dir)

                runstats_id_bull = 'Brams_bully_trained_2hrs'
                runstats_path_bull = '../src/baselines/runstats/' + runstats_id_bull + '/model_weights.ckpt'
                #saver.restore(self.sess, runstats_path_champ)

                #bull_saver.restore(bull_sess, adv_ckpt.model_checkpoint_path)
                bull_saver.restore(self.bull_sess, runstats_path_bull)

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
                                        obs.rpm,
                                        (obs.opponents/-200.0)+1. ))
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
        command = self.champ_agent.act(obs[:29], self.champ_sess)
        # All actions are predicted in [-1, 1]; normalizing back:
        command[1] = (command[1]+1)/2
        command[2] = (command[2]+1)/2
        #command[2] = 0
        command = [command]
        return command

    def drive(self, carstate: State) -> Command:
        #'''
        #Custom Drive Function
        #'''

        #self.swarm_communication(carstate)

        command = Command()

        control_vec = np.full(4, None)
        predicted = self.ddpg_driver(carstate)
        control_vec[2] = predicted[0][0] # steering
        control_vec[0] = predicted[0][1] # accelerator
        control_vec[1] = predicted[0][2] # brake
        control_vec[3] = 1

        print(control_vec)

        #control_vec = self.privilege(control_vec)

        command.accelerator = control_vec[0]
        command.brake = control_vec[1]
        command.steering = control_vec[2]
        command.gear = control_vec[3]

        print("we're on experiment file")

        return command
