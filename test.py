#!python
"""Main module to test the agent training results"""

from environment import UnityEnv
from agent import Agent

def main():
    """Shows training agent results"""

    env = UnityEnv(Agent())

    env.load_actor("actor.pth")
    env.load_critic("critic.pth")
    env.test(100, 1000)

    print("Average Score:", env.avg_score)
    print("Last Score:", env.last_score)


if __name__ == "__main__":
    main()
