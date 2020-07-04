#!python
"""Agent training and testing environment"""

from unityagents import UnityEnvironment
import numpy as np

ENV_PATH = "/Applications/Reacher20.app"

class EnvBase:
    """Base class for environment to unit test its logic"""

    def __init__(self, agent):
        self.__agent = agent
        self.__training_scores = []


    @property
    def training_scores(self):
        return self.__training_scores


    def train(self, max_t, n_episodes):
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


    def test(self, max_t, n_episodes):
        brain = self.__env.brains[self.__brain_name]
        env_info = self.__env.reset()[self.__brain_name]
        num_agents = len(env_info.agents)
        action_size = brain.vector_action_space_size
        states = env_info.vector_observations
        state_size = states.shape[1]

        scores = np.zeros(num_agents)

        while True:
            actions = self.__agent.act(states)

            #actions = np.random.randn(num_agents, action_size)  # select an action (for each agent)
            #actions = np.clip(actions, -1, 1)  # all actions between -1 and 1

            env_info = self.__env.step(actions)[self.__brain_name]  # send all actions to tne environment

            next_states = env_info.vector_observations  # get next state (for each agent)
            rewards = env_info.rewards  # get reward (for each agent)
            dones = env_info.local_done  # see if episode finished
            scores += env_info.rewards  # update the score (for each agent)

            #TODO: agent.step(states, actions, rewards, next_states, dones)

            states = next_states  # roll over states to next time step
            if np.any(dones):  # exit loop if episode finished
                break

        #print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))


class UnityEnv(EnvBase):
    def __init__(self, agent):
        super(UnityEnv, self).__init__(agent)

        self.__env = UnityEnvironment(file_name=ENV_PATH)
        self.__brain_name = self.__env.brain_names[0]
        self.__max_mean_score = 0.0
        self.__avg_score = 0.0
        self.__last_score = 0.0

        info = self.__env.reset(train_mode=False)[self.__brain_name]
        self.__num_agents = len(info.agents)

        brain = self.__env.brains[self.__brain_name]
        self.__action_size = brain.vector_action_space_size

        states = info.vector_observations
        self.__state_size = states.shape[1]


    def __del__(self):
        self.__env.close()


    def load_actor(self, file_path):
        pass # TODO:


    def save_actor(self, file_path):
        pass # TODO:


    def load_critic(self, file_path):
        pass # TODO:


    def save_critic(self, file_path):
        pass # TODO:


    @property
    def num_agents(self):
        return self.__num_agents


    @property
    def action_size(self):
        return self.__action_size


    @property
    def state_size(self):
        return self.__state_size


    @property
    def max_mean_score(self):
        return self.__max_mean_score


    @property
    def avg_score(self):
        return self.__avg_score


    @property
    def last_score(self):
        return self.__last_score


    def _step(self, actions):
        info = self.__env.step(actions)[self.__brain_name]
        return info.vector_observations, info.rewards, info.local_done


    def _reset(self, train_mode):
        info = self.__env.reset(train_mode=train_mode)[self.__brain_name]
        return info.vector_observations
