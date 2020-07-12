#!python
"""Main module to test the agent training results"""

import torch
from env_agent_factory import create_env_agent

def main():
    """Shows training agent results"""

    env, agent = create_env_agent()

    agent.actor_local.load_state_dict(torch.load("actor.pth"))
    agent.critic_local.load_state_dict(torch.load("critic.pth"))

    env.test(agent, 50, 100)

    print("Average Scores:", env.avg_scores)
    print("Last Scores:", env.last_scores)


if __name__ == "__main__":
    main()
