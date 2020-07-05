#!python
"""Main module to test the agent training results"""

from environment import UnityEnv
from agent import Agent

def main():
    """Shows training agent results"""

    env = UnityEnv()

    agent = Agent(env.action_size, env.state_size)
    agent.load_actor("actor.pth")
    agent.load_critic("critic.pth")

    env.test(agent, 50, 100)

    print("Average Scores:", env.avg_scores)
    print("Last Scores:", env.last_scores)


if __name__ == "__main__":
    main()
