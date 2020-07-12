#!python
"""Reinforcement Learning Agent that will train and act in the environment"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from replay_buffer import ReplayBuffer, DEVICE
from neural_nets import Actor, Critic


BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 3e-4        # learning rate of the critic
WEIGHT_DECAY = 0.0001   # L2 weight decay


def _soft_update(local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target

    Params
    ======
        local_model: PyTorch model (weights will be copied from)
        target_model: PyTorch model (weights will be copied to)
        tau (float): interpolation parameter
    """
    for target_param, local_param in zip(target_model.parameters(),
                                         local_model.parameters()):
        target_param.data.copy_(
            tau * local_param.data + (1.0 - tau) * target_param.data)


class Agent:
    """Policy gradient agent to train and act in a distributed environment"""

    # pylint: disable=no-member

    def __init__(self, state_size, action_size):
        """Create an instance of Agent
        :param state_size: state vector dimension
        :param action_size: action vector dimension"""

        self.actor_local = Actor(state_size, action_size, 0)
        self.__actor_target = Actor(state_size, action_size, 0)

        self.__actor_optimizer = optim.Adam(
            self.actor_local.parameters(),
            lr=LR_ACTOR)


        self.critic_local = Critic(state_size, action_size, 0)
        self.__critic_target = Critic(state_size, action_size, 0)

        self.__critic_optimizer = optim.Adam(
            self.critic_local.parameters(),
            lr=LR_CRITIC,
            weight_decay=WEIGHT_DECAY)

        self.__memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, 0)


    def step(self, states, actions, env_info):
        """Performs a training step
        :param states: current states of environments
        :param actions: actions which were taken by the agent upon states.
        :param env_info: Info of agent states after applying actions
        """

        # Save experiences / rewards
        self.__memory.add(states, actions, env_info)

        # Learn, if enough samples are available in memory
        if len(self.__memory) > BATCH_SIZE:
            experiences = self.__memory.sample()
            self.__learn(experiences, GAMMA)


    def act(self, states, add_noise):
        """Calculates action vectors from state vectors for multiple
        environments
        :param states: state vectors from multiple environments
        :param add_noise: if True, adds noise vector
        :return: action vectors for multiple environments"""

        torch_states = torch.from_numpy(states).float().to(DEVICE)

        self.actor_local.eval()

        with torch.no_grad():
            actions = self.actor_local(torch_states).cpu().data.numpy()

        self.actor_local.train()

        if add_noise:
            actions += 0.2 * np.random.randn(*actions.shape)

        return np.clip(actions, -1.0, 1.0)


    def __learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience
        tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done)
            tuples gamma (float): discount factor"""

        states, actions, rewards, next_states, dones = experiences

        # Update critic
        # Get predicted next-state actions and Q values from target models
        actions_next = self.__actor_target(next_states)
        q_targets_next = self.__critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))
        # Compute critic loss
        q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(q_expected, q_targets)
        # Minimize the loss
        self.__critic_optimizer.zero_grad()
        critic_loss.backward()
        self.__critic_optimizer.step()

        # Update actor
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.__actor_optimizer.zero_grad()
        actor_loss.backward()
        self.__actor_optimizer.step()

        # Update target networks
        _soft_update(self.critic_local, self.__critic_target, TAU)
        _soft_update(self.actor_local, self.__actor_target, TAU)
