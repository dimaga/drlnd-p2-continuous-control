#!python
"""environment.py unit tests"""

import unittest
from environment import EnvBase


class AgentStub:
    """Agent stub to test environment"""


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

    def test_create_environment(self):
        env = EnvStub()
        agent = AgentStub()
        env.test(agent)


if __name__ == '__main__':
    unittest.main()
