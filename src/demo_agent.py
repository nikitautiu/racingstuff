import argparse
import logging
import sys

import gym
from gym import wrappers

from gym_torcs.gym_torcs import TorcsEnv


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    args = parser.parse_args()

    # Call `undo_logger_setup` if you want to undo Gym's logger setup
    # and configure things manually. (The default should be fine most
    # of the time.)
    gym.undo_logger_setup()
    logger = logging.getLogger()
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    env = TorcsEnv(throttle=False)
    agent = RandomAgent(env.action_space)

    episode_count = 100
    reward = 0
    done = False

    for _ in range(5):
        done = False
        ob = env.reset()
        while not done:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    env.close()