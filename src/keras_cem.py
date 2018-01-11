import numpy as np
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from rl.agents.cem import CEMAgent
from rl.core import Processor
from rl.memory import EpisodeParameterMemory

from gym_torcs.gym_torcs import TorcsEnv


class ActionDiscretizer(Processor):
    def __init__(self, bins):
        self.bins = bins

    def process_action(self, action):
        """Make from a discrete action, a continuous one"""
        if action.shape == ():
            action = np.array([-1. + action * 2. / self.bins], dtype=np.float32)
        return action

    def process_info(self, info):
        return {}


# Get the environment and extract the number of actions.
env = TorcsEnv(throttle=False, obs_fields=['speedX',
                                           'speedY',
                                           'speedZ',
                                           'trackPos'])

obs_dim = env.observation_space.shape[0]
nb_actions = 16

# Option 1 : Simple model
# model = Sequential()
# model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
# model.add(Dense(10))
# model.add(Activation('softmax'))

# Option 2: deep network
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('softmax'))


print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = EpisodeParameterMemory(limit=1000, window_length=1)

cem = CEMAgent(model=model, nb_actions=nb_actions, memory=memory,
               batch_size=20, nb_steps_warmup=0, train_interval=200, elite_frac=0.1,
               processor=ActionDiscretizer(nb_actions))
cem.compile()
# cem.load_weights('cem_{}_params.h5f'.format('aaa'))
# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
cem.fit(env, nb_steps=100000, visualize=False, verbose=2)

# After training is done, we save the best weights.
# cem.save_weights('cem_{}_params.h5f'.format('aaa'), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
cem.test(env, nb_episodes=5, visualize=False)
