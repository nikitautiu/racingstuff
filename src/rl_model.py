from functools import wraps

import numpy as np
from gym.spaces import Discrete
from keras.callbacks import LambdaCallback

from gym_torcs.gym_torcs import TorcsEnv


def discretize_env(env, bins):
    """Wrapper that discretizes the actions of a torcs env"""
    gear = env.gear_change
    throttle = env.throttle

    old_step = env.step

    @wraps(old_step)
    def step(action):
        obs, rew, done, _ = old_step(process_action(bins, throttle, gear, action))
        return obs, rew, done, {}

    env.action_space = Discrete(bins + bins * int(throttle) + 6 * int(gear))
    env.step = step

    return env


def process_action(bins, throttle, gear, action):
    """Make from a discrete action, a continuous one"""
    if action.shape == ():
        if throttle:
            if action >= bins:
                action_vect = [-1. + (action - bins) * 2. / bins, 0]
            else:
                action_vect = [0, -1. + action * 2. / bins]
        else:
            action_vect = [-1. + action * 2. / bins]
        if gear:
            # the last 7 actions are reserved for gear
            # ofsettedby one, -1 for reverse
            gear_act = max([action - bins * int(throttle) - bins, 0])
            action_vect.append(gear_act)

        return np.array(action_vect, dtype=np.float32)
    return action


class TorcsKerasTrainer(object):
    """Class that encapsulates the behaviour for creating
    ad training KerasRL agents"""

    def __init__(self, obs_fields, throttle=False, gear=False, discrete_actions=5, model_function=None):
        """ Initialize the trainer
        :param obs_fields: the fields to use for the model
        :param throttle: whether the model should have input for throttle
        :param gear: whether the model should have gear input
        :param discrete_actions: how many actions to discretize the continuous input
        :param model_function: the model building function, if any
        """
        self.env = discretize_env(TorcsEnv(obs_fields=obs_fields, throttle=throttle, gear_change=gear),
                                  discrete_actions)

        if model_function is None:
            self.model = self.build_model(self.env.observation_space.shape, self.env.action_space.n)
        else:
            self.model = model_function(self.env.observation_space.shape, self.env.action_space.n)

    def build_model(self, obs_shape, nb_actions):
        """Given the shape of the obseravations and the number of discrete actions,
        return the model(aka a KerasRL agent)"""
        raise NotImplementedError()

    def load_weights(self, filename):
        """Load the weights, from the file, in the model"""
        self.model.load_weights(filename)

    def save_weights(self, *args, **kwargs):
        """Save the weights of the model"""
        self.model.save_weights(*args, **kwargs)

    def fit(self, best_filename, *args, min_step_save=-1, **kwargs):
        """Fit the model ensuring to save the best model periodically
        and log progress"""
        self.rewards_over_eps = []
        self.best_reward = None
        self.best_filename = best_filename
        self.min_step_save = min_step_save

        # call fit with
        self.model.fit(*args, **kwargs, env=self.env, visualize=False,
                       callbacks=[LambdaCallback(on_epoch_end=lambda episode_nb, logs:
                       self.process_episode(episode_nb, logs))] + kwargs.pop('callbacks', []))

    def test(self, *args, **kwargs):
        self.model.test(*args, **kwargs, env=self.env, visualize=False)

    def process_episode(self, episode_nb, episode_log):
        """Check if the episode is better than the best one,
        also append the reward"""
        ep_reward = episode_log['episode_reward']
        nb_steps = episode_log['nb_steps']

        self.rewards_over_eps.append(ep_reward)
        if nb_steps > self.min_step_save and (self.best_reward is None or self.best_reward < ep_reward):
            self.best_reward = ep_reward
            # write the new best episode and overwrite
            self.save_weights(self.best_filename.format(reward=int(ep_reward)), overwrite=True)

            print("\nReward of {} for episode {} better than last -- SAVING\n".format(ep_reward, episode_nb))
