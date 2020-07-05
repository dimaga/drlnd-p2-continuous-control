#!python
"""Reinforcement Learning Agent that will train and act in the environment"""

from replay_buffer import ReplayBuffer
from neural_nets import Actor, Critic

import numpy as np

class Agent:
    """Agent"""

    def __init__(self, action_size, state_size):
        self.__action_size = action_size
        self.__state_size = state_size


    def step(self, states, actions, env_info):
        """Performs a training step"""

        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done


    def act(self, states):
        """Acts upon a state"""

        actions = np.random.randn(states.shape[0], self.__action_size)
        actions = np.clip(actions, -1, 1)
        return actions


    def load_actor(self, file_path):
        pass # TODO:


    def save_actor(self, file_path):
        pass # TODO:


    def load_critic(self, file_path):
        pass # TODO:


    def save_critic(self, file_path):
        pass # TODO:
