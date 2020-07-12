#!python
"""Reinforcement Learning Agent that will train and act in the environment"""

import numpy as np
from replay_buffer import ReplayBuffer
from neural_nets import Actor, Critic


class Agent:
    """Agent"""

    def __init__(self, state_size, action_size):
        self.__state_size = state_size
        self.__action_size = action_size

        self.actor_local = Actor(state_size, action_size, 0)
        self.__actor_target = Actor(state_size, action_size, 0)

        self.critic_local = Critic(state_size, action_size, 0)
        self.__critic_target = Critic(state_size, action_size, 0)


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
