#!python
"""Main module to train the agent"""

import numpy as np
import matplotlib.pyplot as plt
from environment import UnityEnv
from agent import Agent

def main():
    """Trains the agent with an Actor-Critic method"""

    env = UnityEnv(Agent())

    env.train(1000, 1000)

    if env.max_mean_score > 30.0:
        print("Saving actor.pth and critic.pth with score", env.max_mean_score)
        env.save_actor("actor.pth")
        env.save_critic("critic.pth")

    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(len(env.training_scores)), env.training_scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


if __name__ == "__main__":
    main()
