#!python
"""agent.py unit tests"""

import unittest
import numpy as np
from agent import Agent
from environment import InfoStub


class TestAgent(unittest.TestCase):
    """Test cases to verify agent module"""


    def test_agent_act(self):
        """Test how an agent can act"""

        agent = Agent(3, 5, 1)
        states = np.array([[1.0, 2.0, 3.0], [0.3, 2.0, 1.0]])

        actions1 = agent.act(states, False)
        self.assertEqual((2, 5), actions1.shape)

        actions2 = agent.act(states, False)
        self.assertTrue(np.allclose(actions2, actions1))

        actions3 = agent.act(states, True)
        self.assertFalse(np.allclose(actions2, actions3))


    def test_agent_step(self):
        """Train the agent to approach a goal in 1D world. state[0] is agent
        position, state[1] is a goal position, action[0] is a displacement
        of an agent to the goal"""

        state_size = 2
        action_size = 1
        environments = 20

        agent = Agent(state_size, action_size, environments)
        np.random.seed(0)

        at_least_one_reached = False
        n_episodes = 150

        for episode in range(n_episodes):

            states = np.random.random((environments, state_size))

            at_least_one_reached = False

            agent.reset()

            for _ in range(10):
                train_step = episode < n_episodes - 1
                actions = agent.act(states, train_step)

                info = InfoStub()
                info.vector_observations = states.copy()
                info.vector_observations[:, 0] += actions.reshape(-1)

                info.rewards = np.exp(-np.abs(
                    info.vector_observations[:, 0]
                    - info.vector_observations[:, 1]))

                info.local_done = info.rewards > 0.999

                if train_step:
                    agent.step(states, actions, info)

                if np.any(info.local_done):
                    at_least_one_reached = True
                    break

                states = info.vector_observations

        self.assertTrue(at_least_one_reached)



if __name__ == '__main__':
    unittest.main()
