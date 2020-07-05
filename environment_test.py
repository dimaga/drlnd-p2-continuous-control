#!python
"""environment.py unit tests"""

import unittest
from environment import EnvBase


class AgentStub:
    """Agent stub to test environment"""

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



class EnvStub(EnvBase):

    @property
    def num_agents(self):
        return 2


    @property
    def action_size(self):
        return 5


    @property
    def state_size(self):
        return 3


    def _step(self, actions):
        return None


    def _reset(self, train_mode):
        return None


class TestEnvironment(unittest.TestCase):
    """Test cases to verify environment module"""

    def test_environment_test(self):
        env = EnvStub(AgentStub())
        env.test(10, 20)


    #def test_environment_train(self):
    #    env = EnvStub(AgentStub())
    #    env.train(10, 20)


if __name__ == '__main__':
    unittest.main()
