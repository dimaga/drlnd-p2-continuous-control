#!python
"""Agent training and testing environment"""

from abc import ABC, abstractmethod
import numpy as np
from unityagents import UnityEnvironment


ENV_PATH = "/Applications/Reacher20.app"


class EnvBase(ABC):
    """Environment logic that does not depend on Unity"""

    def __init__(self, num_agents):
        """Create environment for testing and training with

        :param num_agents: Number of copies of the environment for distributed
        training
        """

        self.__num_agents = num_agents
        self.__total_scores = []
        self.__max_mean_scores = np.zeros((num_agents, ))
        self.__avg_scores = np.zeros((num_agents, ))
        self.__last_scores = np.zeros((num_agents, ))

        self.report_progress = lambda episode, mean_score: None


    @property
    def total_scores(self):
        """:return: History of total scores of each environment instance per
        each episode"""
        return self.__total_scores


    @property
    def max_mean_scores(self):
        """:return: Maximum mean score of 100-episode window per each
        environment instance"""
        return self.__max_mean_scores


    @property
    def avg_scores(self):
        """:return: Average scores of all episodes per each environment
        instance"""
        return self.__avg_scores


    @property
    def last_scores(self):
        """:return: Last episode scores of all episodes per each environment
        instance"""
        return self.__last_scores


    @property
    def num_agents(self):
        """:return: Number of environment instances which are trained and tested
        in parallel"""
        return self.__num_agents


    @property
    @abstractmethod
    def action_size(self):
        """:return: Action vector size"""
        raise NotImplementedError


    @property
    @abstractmethod
    def state_size(self):
        """:return: State vector size"""
        raise NotImplementedError


    @abstractmethod
    def _step(self, actions):
        """Apply actions to environment instances to update their states. The
        method must be overloaded by a subclass
        :param actions: action vectors
        :return: New state vectors per each environment instance"""
        raise NotImplementedError


    @abstractmethod
    def _reset(self, train_mode):
        """Start training or testing in environment instances. The method must
        be overloaded by a subclass
        :param train_mode: True if training mode.
        :return: Initial state vectors per each environment instance"""
        raise NotImplementedError


    def train(self, agent, max_t, n_episodes):
        """Train agent by running num_agent environments for n_episodes. The
        training algorithm will pick the best results over 100-episode window
        and re-store it in agent.actor_local and agent.critic_local
        :param agent: Agent, implementing neural networks and training
        algorithms
        :param max_t: Maximum number of steps per episode if it is not finished
        earlier
        :param n_episodes: Number of episodes for training
        """
        self.__run(True, agent, max_t, n_episodes)


    def test(self, agent, max_t, n_episodes):
        """ Test agent by running num_agent environments for n_episodes.
        :param agent: Agent, implementing neural networks and training
        algorithms
        :param max_t: Maximum number of steps per episode if it is not finished
        earlier
        :param n_episodes: Number of episodes for training
        """
        self.__run(False, agent, max_t, n_episodes)


    def __run(self, train_mode, agent, max_t, n_episodes):

        self.__total_scores = []
        self.__max_mean_scores.fill(0.0)

        actor = None
        critic = None

        scores = np.array([0.0] * self.num_agents)

        for episode in range(n_episodes):
            states = self._reset(train_mode)

            scores.fill(0)

            for _ in range(max_t):
                actions = agent.act(states)

                info = self._step(actions)

                if train_mode:
                    agent.step(states, actions, info)

                scores += info.rewards
                states = info.vector_observations

                episode_finished = np.any(info.local_done)
                if np.any(episode_finished):
                    break

            self.__total_scores.append(scores.copy())

            mean_scores = np.mean(self.__total_scores[-100:], axis=0)
            self.report_progress(episode, mean_scores)

            if self.__max_mean_scores.mean() < mean_scores.mean():
                self.__max_mean_scores = mean_scores

                if train_mode:
                    actor = agent.actor_local.state_dict().copy()
                    critic = agent.critic_local.state_dict().copy()

        if train_mode and actor is not None and critic is not None:
            agent.actor_local.load_state_dict(actor)
            agent.critic_local.load_state_dict(critic)

        self.__avg_scores = np.mean(self.__total_scores, axis=0)
        self.__last_scores = scores


class UnityEnv(EnvBase):
    """Unity-based environment for training and testing agents"""

    def __init__(self):
        env = UnityEnvironment(file_name=ENV_PATH)
        brain_name = env.brain_names[0]
        info = env.reset(train_mode=False)[brain_name]

        super(UnityEnv, self).__init__(len(info.agents))

        self.report_progress = \
            lambda episode, mean_scores: print(
                '\rEpisode {}\tAverage Score: {:.2f} '.format(
                    episode + 1,
                    mean_scores.mean()),
                end="" if (episode + 1) % 100 != 0 else "\n")

        self.__env = env
        self.__brain_name = brain_name

        brain = self.__env.brains[self.__brain_name]
        self.__action_size = brain.vector_action_space_size

        states = info.vector_observations
        self.__state_size = states.shape[1]


    def __del__(self):
        self.__env.close()


    @property
    def action_size(self):
        return self.__action_size


    @property
    def state_size(self):
        return self.__state_size


    def _step(self, actions):
        info = self.__env.step(actions)[self.__brain_name]
        return info


    def _reset(self, train_mode):
        info = self.__env.reset(train_mode=train_mode)[self.__brain_name]
        return info.vector_observations
