import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.core import Processor
from rl.policy import *
from rl.memory import SequentialMemory

from gym_torcs.gym_torcs import TorcsEnv

class ActionDiscretizer(Processor):
    def __init__(self, bins, throttle=False):
        self.throttle = throttle
        self.bins = bins

    def process_action(self, action):
        """Make from a discrete action, a continuous one"""
        if action.shape == ():
            if self.throttle:
                if action > self.bins:
                    action = np.array([-1. + (action-self.bins) * 2. / self.bins, 0], dtype=np.float32)
                else:
                    action = np.array([0, -1. + action * 2. / self.bins], dtype=np.float32)

            else:
                action = np.array([-1. + action * 2. / self.bins], dtype=np.float32)
            return action

        return action

    def process_info(self, info):
        return {}


# Get the environment and extract the number of actions.
env = TorcsEnv(throttle=True, obs_fields=['speedX',
                                           'speedY',
                                           'speedZ',
                                           'track',
                                           'focus',
                                           'trackPos',
                                           'angle'])


# Get the environment and extract the number of actions.
np.random.seed(123)
nb_actions = 32

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = MaxBoltzmannQPolicy()


dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=2000,
               target_model_update=1e-2, policy=policy, processor=ActionDiscretizer(16, True))
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.load_weights('dqn_with_throttle_backyard_{}_weights.h5f'.format('dqn'))
# dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)

# After training is done, we save the final weights.
# dqn.save_weights('dqn_with_throttle_backyard_{}_weights.h5f'.format('dqn'), overwrite=True)
# dqn.load_weights('dqn_{}_weights.h5f'.format('dqn'))

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=False)