#!python
"""Main module to test the agent training results"""

import torch
from environment import UnityEnv
from agent import Agent

def main():
    """Shows training agent results"""

    env = UnityEnv()

    agent = Agent(env.action_size, env.state_size)
    agent.actor_local.load_state_dict(torch.load("actor.pth"))
    agent.critic_local.load_state_dict(torch.load("critic.pth"))

    env.test(agent, 50, 100)

    print("Average Scores:", env.avg_scores)
    print("Last Scores:", env.last_scores)


if __name__ == "__main__":
    main()
