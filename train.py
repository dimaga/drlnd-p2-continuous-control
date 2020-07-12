#!python
"""Main module to train the agent"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from env_agent_factory import create_env_agent

def main():
    """Trains the agent with an Actor-Critic method"""

    env, agent = create_env_agent()

    env.train(agent, 50, 1000)

    if np.all(env.max_mean_scores > 30.0):

        print(
            "Saving actor.pth and critic.pth with score",
            env.max_mean_scores)

        torch.save(agent.actor_local.state_dict(), "actor.pth")
        torch.save(agent.critic_local.state_dict(), "critic.pth")
    else:

        print(
            "Some of the scores are below 30.0, not saved",
            env.max_mean_scores)

    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(len(env.total_scores)), env.total_scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


if __name__ == "__main__":
    main()
