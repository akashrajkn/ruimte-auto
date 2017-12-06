import gym
from gym import spaces
import numpy as np
# from os import path
import baselines.ddpg.snakeoil3_gym as snakeoil3
import numpy as np
import copy
import collections as col
import os
import time
from random import randint


class TorcsEnv:
    terminal_judge_start = 2000  # If after 100 timestep still no progress, terminated
    termination_limit_progress = 5  # [km/h], episode terminates if car is running slower than this limit
    default_speed = 50

    initial_reset = True

    def __init__(self, vision=False, throttle=False, gear_change=False):
        self.vision = vision
        self.throttle = throttle
        self.gear_change = gear_change

        self.initial_run = True

        ##print("launch torcs")
        os.system('pkill torcs')
        time.sleep(0.5)

        xml_path = os.path.dirname(os.path.abspath(__file__)) + '/../../../resources/xmls/'
        track_id = (randint(0, len(os.listdir(xml_path))-1))
        track_xml = os.listdir(xml_path)[track_id]
        os.system('torcs -r ' + xml_path + track_xml + ' -nofuel -nolaptime &')
        #os.system('torcs -r ~/practice_results_mode.xml -nofuel -nolaptime &')
        time.sleep(0.5)
        #os.system('sh baselines/ddpg/autostart.sh')
        #time.sleep(0.5)

        """
        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=3101, vision=self.vision)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        """
        if throttle is False:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        else:
            #self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,))


        high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf])
        low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf])
        #self.observation_space = spaces.Box(low=low, high=high)
        self.observation_space = spaces.Box(low=low, high=high)


    def step(self, u):
       #print("Step")
        # convert thisAction to the actual torcs actionstr




        client = self.client

        # Check if client is off
        if self.client.ison is False:
            print('RESETTING TORCS')
            self.reset()
            self.client.ison = True

        this_action = self.agent_to_torcs(u)

        # Apply Action
        action_torcs = client.R.d

        # Steering
        action_torcs['steer'] = this_action['steer']  # in [-1, 1]

        #  Simple Autnmatic Throttle Control by Snakeoil
        if self.throttle is False:
            target_speed = self.default_speed
            if client.S.d['speedX'] < target_speed - (client.R.d['steer']*50):
                client.R.d['accel'] += .01
            else:
                client.R.d['accel'] -= .01

            if client.R.d['accel'] > 0.2:
                client.R.d['accel'] = 0.2

            if client.S.d['speedX'] < 10:
                client.R.d['accel'] += 1/(client.S.d['speedX']+.1)

            # Traction Control System
            if ((client.S.d['wheelSpinVel'][2]+client.S.d['wheelSpinVel'][3]) -
               (client.S.d['wheelSpinVel'][0]+client.S.d['wheelSpinVel'][1]) > 5):
                action_torcs['accel'] -= .2
        else:
            action_torcs['accel'] = this_action['accel']
            action_torcs['brake'] = this_action['brake']

        #  Automatic Gear Change by Snakeoil
        if self.gear_change is True:
            action_torcs['gear'] = this_action['gear']
        else:
            #  Automatic Gear Change by Snakeoil is possible
            action_torcs['gear'] = 1
            if self.throttle:
                if client.S.d['speedX'] > 50:
                    action_torcs['gear'] = 2
                if client.S.d['speedX'] > 80:
                    action_torcs['gear'] = 3
                if client.S.d['speedX'] > 110:
                    action_torcs['gear'] = 4
                if client.S.d['speedX'] > 140:
                    action_torcs['gear'] = 5
                if client.S.d['speedX'] > 170:
                    action_torcs['gear'] = 6
        # Save the privious full-obs from torcs for the reward calculation
        obs_pre = copy.deepcopy(client.S.d)

        # One-Step Dynamics Update #################################
        # Apply the Agent's action into torcs
        client.respond_to_server()
        # Get the response of TORCS
        client.get_servers_input()

        # Get the current full-observation from torcs
        obs = client.S.d

        # Make an obsevation from a raw observation vector from TORCS
        self.observation = self.make_observaton(obs)

        # Reward setting Here #######################################
        # direction-dependent positive reward
        track = np.array(obs['track'])
        trackPos = np.array(obs['trackPos'])
        sp = np.array(obs['speedX'])
        damage = np.array(obs['damage'])
        rpm = np.array(obs['rpm'])

        progress = sp*np.cos(obs['angle']) - np.abs(sp*np.sin(obs['angle'])) - sp*np.abs(obs['trackPos'])
        reward = progress

        # collision detection
        if obs['damage'] - obs_pre['damage'] > 0:
            reward = -10

        # Termination judgement #########################
        episode_terminate = False
        if (abs(track.any()) > 1 or abs(trackPos) > 1):  # Episode is terminated if the car is out of track
            reward = reward - 40
            episode_terminate = True
            client.R.d['meta'] = True


        if self.terminal_judge_start < self.time_step: # Episode terminates if the progress of agent is small
            if progress < self.termination_limit_progress:
                #print("No progress")
                episode_terminate = True
                client.R.d['meta'] = True

        if np.cos(obs['angle']) < 0: # Episode is terminated if the agent runs backward
            reward = -100
            episode_terminate = True
            client.R.d['meta'] = True


        if client.R.d['meta'] is True: # Send a reset signal
            self.initial_run = False
            #print(self.get_lap_info())
            client.respond_to_server()

        self.time_step += 1

        #if reward<0:
        #    reward = -1

        #print('reward: ', reward)
        #return self.get_obs(), reward, client.R.d['meta'], {}
        return self.get_state(), reward, client.R.d['meta'], {}

    def reset(self, relaunch=False):
        #print("Reset")

        self.time_step = 0

        if self.initial_reset is not True:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

            ## TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
            if relaunch is True:
                self.reset_torcs()
                #print("### TORCS is RELAUNCHED ###")

        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=3001, vision=self.vision)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        self.observation = self.make_observaton(obs)

        self.last_u = None

        self.initial_reset = False
        #return self.get_obs()
        return self.get_state()

    def close(self):
        #print("### CLOSING TORCS ###")
        os.system('pkill torcs')


    def get_lap_info(self):
        return (obs['curLapTime'], obs['lastLapTime'])


    def get_obs(self):
        return self.observation

    def get_state(self):
        return self.make_state(self.observation)

    def reset_torcs(self):
       #print("relaunch torcs")
        os.system('pkill torcs')
        #time.sleep(0.5)

        # path = os.path.dirname(os.path.abspath(__file__)) + '/../../../resources/xmls/'
        # for filename in os.listdir(path):
        #     config_file = path + filename
        #
        #     print('-----------'+ config_file +'------------')
        #     os.system('torcs -r ' + config_file + ' -nofuel -nolaptime &')
        xml_path = os.path.dirname(os.path.abspath(__file__)) + '/../../../resources/xmls/'
        track_id = (randint(0, len(os.listdir(xml_path))-1))
        track_xml = os.listdir(xml_path)[track_id]
        os.system('torcs -r ' + xml_path + track_xml + ' -nofuel -nolaptime &')
        #os.system('torcs -r ~/practice_results_mode.xml -nofuel -nolaptime &')

        #time.sleep(0.5)
        #os.system('sh baselines/ddpg/autostart.sh')
        #time.sleep(0.5)

    def agent_to_torcs(self, u):
        torcs_action = {'steer': u[0]}

        if self.throttle is True:  # throttle action is enabled

            u[1] = (u[1]+1)/2
            u[2] = (u[2]+1)/2
            #u[2] = 0 #disable brake
            torcs_action.update({'accel': u[1]})
            torcs_action.update({'brake': u[2]})

        if self.gear_change is True: # gear change action is enabled
            torcs_action.update({'gear': int(u[3])})

        return torcs_action


    def make_observaton(self, raw_obs):
        names = ['focus',
                 'speedX', 'speedY', 'speedZ', 'angle', 'damage',
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


    def make_state(self, obs):
        return np.hstack((obs.angle, obs.track, obs.trackPos, obs.speedX, obs.speedY,  obs.speedZ, obs.wheelSpinVel/100.0, obs.rpm))
