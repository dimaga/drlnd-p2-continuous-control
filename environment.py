#!python
"""Agent training and testing environment"""

from abc import ABC, abstractmethod
import numpy as np
from unityagents import UnityEnvironment


ENV_PATH = "/Applications/Reacher20.app"


class EnvBase(ABC):
    """Environment logic that does not depend on Unity"""

    def __init__(self, num_agents):
        self.__num_agents = num_agents
        self.__training_scores = []
        self.__max_mean_scores = np.zeros((num_agents, ))
        self.__avg_scores = np.zeros((num_agents, ))
        self.__last_scores = np.zeros((num_agents, ))


    @property
    def training_scores(self):
        return self.__training_scores


    @property
    def max_mean_scores(self):
        return self.__max_mean_scores


    @property
    def avg_scores(self):
        return self.__avg_scores


    @property
    def last_scores(self):
        return self.__last_scores


    @property
    def num_agents(self):
        return self.__num_agents


    @property
    @abstractmethod
    def action_size(self):
        raise NotImplementedError


    @property
    @abstractmethod
    def state_size(self):
        raise NotImplementedError


    @abstractmethod
    def _step(self, actions):
        raise NotImplementedError


    @abstractmethod
    def _reset(self, train_mode):
        raise NotImplementedError


    def train(self, agent, max_t, n_episodes):
        """
        brain = self.__env.brains[self.__brain_name]
        env_info = self.__env.reset()[self.__brain_name]
        num_agents = len(env_info.agents)
        action_size = brain.vector_action_space_size
        states = env_info.vector_observations
        state_size = states.shape[1]

        scores = np.zeros(num_agents)

        scores_deque = deque(maxlen=print_every)
        scores = []
        for i_episode in range(1, n_episodes + 1):
            state = env.reset()
            agent.reset()
            score = 0
            for t in range(max_t):
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            scores_deque.append(score)
            scores.append(score)
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode,
                                                               np.mean(
                                                                   scores_deque)),
                  end="")
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            if i_episode % print_every == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode,
                                                                   np.mean(
                                                                       scores_deque)))

        while True:
            actions = agent.act(states)

            env_info = self.__env.step(actions)[self.__brain_name]

            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            scores += env_info.rewards

            agent.step(states, actions, rewards, next_states, dones)

            states = next_states
            if np.any(dones):
                break

        """

        # print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))


    def test(self, agent, max_t, n_episodes):

        total_scores = []
        for episode in range(n_episodes):
            states = self._reset(False)

            scores = np.array([0.0] * self.num_agents)

            print("episode", episode)

            for step in range(max_t):
                actions = agent.act(states)

                info = self._step(actions)
                scores += info.rewards

                states = info.vector_observations

                episode_finished = np.any(info.local_done)
                if np.any(episode_finished):
                    break

            total_scores.append(scores)
            self.__avg_scores = np.mean(total_scores, axis=1)
            self.__last_scores = scores


class UnityEnv(EnvBase):
    def __init__(self):
        env = UnityEnvironment(file_name=ENV_PATH)
        brain_name = env.brain_names[0]
        info = env.reset(train_mode=False)[brain_name]

        super(UnityEnv, self).__init__(len(info.agents))

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
