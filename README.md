# AI Racing Project

## Abstract
Given a simulation of racing tracks and agents controlling realistic cars on the track, this project
aims to train some agents to finish them in the least amount of time. Different models are to be used
to control the agents based on inputs from the simulation. After training on the same simulation, the
performance of the models will be compared in terms of lap times.


## Resources
* [Intro to deep reinforcement learning](https://lopespm.github.io/machine_learning/2016/10/06/deep-reinforcement-learning-racing-game.html)
* [GYM - a reinforcement learning framework](https://github.com/openai/gym)
* [University of Toronto - Paper on Q-Learning for Atari games](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
* [Deep Q-learning implementation for the racing environment](https://gym.openai.com/evaluations/eval_BPzPoiBtQOCj8yItyHLhmg/)
* [Simulated Car Racing Championship Competition](https://arxiv.org/pdf/1304.1672.pdf)
* [Evolving Competitive Car Controllers for Racing Games with Neuroevolution](https://archive.alvb.in/msc/04_infoea/seminar/papers/NEATtorcs.pdf)
  
We will be try to use the environments in a similar manner to the Q-learning implementation. The gym builtin one will be used as toy enviorment, but the end goal is to interface it with the [TORCS simulator](https://github.com/ugo-nama-kun/gym_torcs). To accomplish this, we intend to use [this implementaion](https://github.com/ugo-nama-kun/gym_torcs) of a TORC environment or a fork of it to suit our needs.
