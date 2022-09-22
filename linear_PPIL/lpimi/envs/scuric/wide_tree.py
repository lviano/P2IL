"""Python Script Template."""
from collections import defaultdict

from rllib.environment.mdp import MDP


class WideTree(MDP):
    """Implementation of a WdideTree Problem."""

    def __init__(self, reward=1.0):
        num_states = 11
        num_actions = 2
        transitions = self._build_mdp(reward=reward)
        self.check_transitions(transitions, num_states, num_actions)
        initial_state = 0
        super().__init__(transitions, num_states, num_actions, initial_state)

    @staticmethod
    def _build_mdp(reward):
        """Build the transition dictionary."""
        transitions = defaultdict(list)
        transitions[(0, 0)].append({"next_state": 1, "probability": 1.0, "reward": 0})
        transitions[(0, 1)].append({"next_state": 2, "probability": 1.0, "reward": 0})

        for i in range(2):
            transitions[(1, 0)].append(
                {"next_state": 3 + i, "probability": 0.5, "reward": 0}
            )
            transitions[(1, 1)].append(
                {"next_state": 5 + i, "probability": 0.5, "reward": 0}
            )
            transitions[(2, 0)].append(
                {"next_state": 7 + i, "probability": 0.5, "reward": reward}
            )
            transitions[(2, 1)].append(
                {"next_state": 9 + i, "probability": 0.5, "reward": reward}
            )
        for j in range(8):
            for a in range(2):
                transitions[(3 + j, a)].append(
                    {"next_state": 0, "probability": 1.0, "reward": 0}
                )
        return transitions


if __name__ == "__main__":
    from rllib.environment import GymEnvironment
    from rllib.environment.utilities import transitions2kernelreward

    import qreps  # noqa: F401

    env = GymEnvironment("WideTree-v0", reward=1)
    kernel, reward = transitions2kernelreward(
        env.env.transitions, env.num_states, env.num_actions
    )
    print(kernel, reward)
    state = env.reset()
    print(state)
    for i in range(10):
        action = env.action_space.sample()
        next_state, r, done, f = env.step(action)
        print(state, action, next_state, r, done)
        state = next_state
