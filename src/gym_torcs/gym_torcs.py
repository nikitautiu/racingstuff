import copy
import os

import gym
import numpy as np
from gym import spaces

# from os import path
import gym_torcs.snakeoil3_gym as snakeoil3

# default observation fields

DEFAULT_FIELDS = ['focus',
                  'speedX',
                  'speedY',
                  'speedZ',
                  'opponents',
                  'rpm',
                  'track',
                  'wheelSpinVel',
                  'angle']

# intervals for the values
OBS_SPACE_DEF = {
    'focus': (0., 1., (5,)),
    'speedX': (-np.inf, np.inf, (1,)),
    'speedY': (-np.inf, np.inf, (1,)),
    'speedZ': (-np.inf, np.inf, (1,)),
    'opponents': (0., 1., (36,)),
    'rpm': (0., np.inf, (1,)),
    'track': (0., 1., (19,)),
    'wheelSpinVel': (0., np.inf, (4,)),
    'angle': (-np.pi, np.pi, (1,)),
    'distFromStart': (0., np.inf, (1,)),
    'trackPos': (-1., 1., (1,)),
    'gear': (-1., 6., (1,))
}


def make_observation_space(obs_fields):
    """Given a list of fields toa be included in the observation, generate the observation space"""
    definitions = [OBS_SPACE_DEF[field] for field in obs_fields]
    return gym.spaces.Box(low=np.concatenate([np.full(definition[2], definition[0], dtype=np.float32)
                                              for definition in definitions]),
                          high=np.concatenate([np.full(definition[2], definition[1], dtype=np.float32)
                                               for definition in definitions]))


class TorcsEnv(object):
    terminal_judge_start = 500  # Speed limit is applied after this step
    termination_limit_progress = 5  # [km/h], episode terminates if car is running slower than this limit
    default_speed = 50

    initial_reset = True

    def __init__(self, throttle=False, gear_change=False, obs_fields=DEFAULT_FIELDS, reset_on_damage=True):
        # print("Init")
        self.reset_on_damage = reset_on_damage
        self.obs_fields = obs_fields
        self.vision = 'img' in self.obs_fields
        self.throttle = throttle
        self.gear_change = gear_change

        self.initial_run = True

        self.create_client()

        if throttle is False:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

        self.observation_space = make_observation_space(self.obs_fields)

    def step(self, u):
        # print("Step")
        # convert thisAction to the actual torcs actionstr
        client = self.client

        this_action = self.agent_to_torcs(u)

        # Apply Action
        action_torcs = client.R.d

        # Steering
        action_torcs['steer'] = this_action['steer']  # in [-1, 1]

        #  Simple Autnmatic Throttle Control by Snakeoil
        if self.throttle is False:
            target_speed = self.default_speed
            if client.S.d['speedX'] < target_speed - (client.R.d['steer'] * 50):
                client.R.d['accel'] += .01
            else:
                client.R.d['accel'] -= .01

            if client.R.d['accel'] > 0.2:
                client.R.d['accel'] = 0.2

            if client.S.d['speedX'] < 10:
                client.R.d['accel'] += 1 / (client.S.d['speedX'] + .1)

            # Traction Control System
            if ((client.S.d['wheelSpinVel'][2] + client.S.d['wheelSpinVel'][3]) -
                    (client.S.d['wheelSpinVel'][0] + client.S.d['wheelSpinVel'][1]) > 5):
                action_torcs['accel'] -= .2
        else:
            action_torcs['accel'] = this_action['accel']

            # Automatic Gear Change by Snakeoil
        if self.gear_change is True:
            action_torcs['gear'] = this_action['gear']
        else:
            #  Automatic Gear Change by Snakeoil is possible
            action_torcs['gear'] = 1

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

        done, reward = self._compute_reward_termination(obs, obs_pre)
        done = done or (
        self.time_step + 10 > self.client.maxSteps)  # a bit of leeway, prevents the server shutting down

        # decide whether to stop the simulation
        client.R.d['meta'] = done
        if client.R.d['meta'] is True:  # Send a reset signal
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1

        # return the raw obs as the last param
        return self.get_obs(), reward, client.R.d['meta'], obs

    def _compute_reward_termination(self, obs, obs_pre):
        """Compute the reward and wether the simulation is really done"""
        # Reward setting Here #######################################
        # direction-dependent positive reward
        track = np.array(obs['track'])
        sp = np.array(obs['speedX'])
        progress = sp * np.cos(obs['angle'])
        reward = progress
        # collision detection

        done = False
        if obs['damage'] - obs_pre['damage'] > 0:
            reward = -1000
            if self.reset_on_damage:
                done = True

        # Termination judgement #########################
        if track.min() < 0:  # Episode is terminated if the car is out of track
            done = True
            reward = -1000

        if self.terminal_judge_start < self.time_step:  # Episode terminates if the progress of agent is small
            if progress < self.termination_limit_progress:
                done = True
                reward = -1000

        if np.cos(obs['angle']) < 0:  # Episode is terminated if the agent runs backward
            done = True
            reward = -1000

        return done, reward

    def reset(self, relaunch=False):
        # print("Reset")

        self.time_step = 0

        if self.initial_reset is not True:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

            ## TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
            if relaunch is True:
                self.reset_torcs()
                print("### TORCS is RELAUNCHED ###")

        self.create_client()

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        self.observation = self.make_observaton(obs)

        self.last_u = None

        self.initial_reset = False
        return self.get_obs()

    def create_client(self, start_game=False):
        """Creates a client with the given settings"""
        self.client = snakeoil3.Client(p=3101, vision=self.vision, start_torcs=start_game)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = 100000

    def close(self):
        self.end()

    def end(self):
        os.system('pkill torcs')

    def get_obs(self):
        return self.observation

    def reset_torcs(self):
        self.client.restart_game()

    def agent_to_torcs(self, u):
        torcs_action = {'steer': u[0]}

        if self.throttle:  # throttle action is enabled
            torcs_action.update({'accel': u[1]})

        if self.gear_change:  # gear change action is enabled
            torcs_action.update({'gear': u[2]})

        return torcs_action

    def obs_vision_to_image_rgb(self, obs_image_vec):
        image_vec = obs_image_vec
        rgb = []
        temp = []
        # convert size 64x64x3 = 12288 to 64x64=4096 2-D list
        # with rgb values grouped together.
        # Format similar to the observation in openai gym
        for i in range(0, 12286, 3):
            temp.append(image_vec[i])
            temp.append(image_vec[i + 1])
            temp.append(image_vec[i + 2])
            rgb.append(temp)
            temp = []
        return np.array(rgb, dtype=np.uint8)

    def make_observaton(self, raw_obs):
        processed_vals = dict(focus=np.array(raw_obs['focus'], dtype=np.float32) / 200.,
                              speedX=np.array([raw_obs['speedX']], dtype=np.float32) / self.default_speed,
                              speedY=np.array([raw_obs['speedY']], dtype=np.float32) / self.default_speed,
                              speedZ=np.array([raw_obs['speedZ']], dtype=np.float32) / self.default_speed,
                              opponents=np.array(raw_obs['opponents'], dtype=np.float32) / 200.,
                              rpm=np.array([raw_obs['rpm']], dtype=np.float32) / 5000,
                              track=np.array(raw_obs['track'], dtype=np.float32) / 200.,
                              wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
                              angle=np.array([raw_obs['angle']], dtype=np.float32),
                              distFromStart=np.array([raw_obs['distFromStart']], dtype=np.float32),
                              trackPos=np.array([raw_obs['trackPos']], dtype=np.float32),
                              gear=np.array([raw_obs['gear']], dtype=np.float32) / 6)

        if self.vision:
            # if image is specified, add it, otherwise no, too expensive to compute
            image_rgb = self.obs_vision_to_image_rgb(raw_obs['img'])

            processed_vals['img'] = image_rgb

        # only return the needed fields
        observations = [processed_vals[name] for name in self.obs_fields]
        return np.concatenate(observations)
