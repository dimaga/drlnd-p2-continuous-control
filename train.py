#!python
"""Main module to train the agent"""

import numpy as np
import matplotlib.pyplot as plt
from environment import UnityEnv
from agent import Agent

def main():
    """Trains the agent with an Actor-Critic method"""


    env = UnityEnv()

    agent = Agent(env.action_size, env.state_size)
    env.train(agent, 50, 1000)

    if np.all(env.max_mean_scores > 30.0):

        print(
            "Saving actor.pth and critic.pth with score",
            env.max_mean_scores)

        agent.save_actor("actor.pth")
        agent.save_critic("critic.pth")
    else:

        print(
            "Some of the scores are below 30.0, not saved",
            env.max_mean_scores)

    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(len(env.training_scores)), env.training_scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


if __name__ == "__main__":
    main()
