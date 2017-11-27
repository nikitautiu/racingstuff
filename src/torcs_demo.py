from gym_torcs.gym_torcs import TorcsEnv
import numpy as np


class Agent(object):
    def __init__(self, dim_action):
        self.dim_action = dim_action

    def act(self, ob, reward, done, vision_on=False):
        print(ob)

        # Get an Observation from the environment.
        # Each observation vectors are numpy array.
        # focus, opponents, track sensors are scaled into [0, 1]. When the agent
        # is out of the road, sensor variables return -1/200.
        # rpm, wheelSpinVel are raw values and then needed to be preprocessed.
        # vision is given as a tensor with size of (64*64, 3) = (4096, 3) <-- rgb
        # and values are in [0, 255]
        if vision_on is False:
            focus, speedX, speedY, speedZ, opponents, rpm, track, wheelSpinVel, raw = ob
        else:
            focus, speedX, speedY, speedZ, opponents, rpm, track, wheelSpinVel, vision, raw = ob

            """ The code below is for checking the vision input. This is very heavy for real-time Control
                So you may need to remove.

            img = np.ndarray((64,64,3))
            for i in range(3):
                img[:, :, i] = 255 - vision[:, i].reshape((64, 64))

            plt.imshow(img, origin='lower')
            plt.draw()
            plt.pause(0.001)
            """
        return np.tanh(np.random.randn(self.dim_action)) # random action


def main():
    # Generate a Torcs environment
    # enable vision input, the action is steering only (1 dim continuous action)
    env = TorcsEnv(vision=False, throttle=False)

    # without vision input, the action is steering and throttle (2 dim continuous action)
    # env = TorcsEnv(vision=False, throttle=True)

    ob = env.reset(relaunch=False)  # with torcs relaunch (avoid memory leak bug in torcs)
    # ob = env.reset()  # without torcs relaunch

    # Generate an agent
    agent = Agent(1)  # steering only
    for i in range(10):
        ob = env.reset(relaunch=False)  # with torcs relaunch (avoid memory leak bug in torcs)
        for _ in range(100):
            action = agent.act(ob, None, None, vision_on=False)

            # single step
            ob, reward, done, _ = env.step(action)

    # shut down torcs
    env.end()


if __name__ == '__main__':
    main()