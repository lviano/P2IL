"""Python Script Template."""
from collections import defaultdict

from rllib.environment.mdp import MDP


class RiverSwim(MDP):
    """Implementation of a WdideTree Problem."""

    def __init__(self, length=5):
        num_states = length + 1
        num_actions = 2
        transitions = self._build_mdp(length=length)
        self.check_transitions(transitions, num_states, num_actions)
        initial_state = 0
        super().__init__(transitions, num_states, num_actions, initial_state)

    @staticmethod
    def _build_mdp(length):
        """Build the transition dictionary."""
        transitions = defaultdict(list)
        transitions[(0, 1)].append(
            {"next_state": 0, "probability": 1.0, "reward": 5 / 1000}
        )
        transitions[(0, 0)].append({"next_state": 0, "probability": 0.1, "reward": 0})
        transitions[(0, 0)].append({"next_state": 1, "probability": 0.9, "reward": 0})

        for j in range(1, length + 1):
            transitions[(j, 1)].append(
                {"next_state": j - 1, "probability": 1.0, "reward": 0}
            )
        for j in range(1, length):
            transitions[(j, 0)].append(
                {"next_state": j - 1, "probability": 0.05, "reward": 0}
            )
            transitions[(j, 0)].append(
                {"next_state": j, "probability": 0.05, "reward": 0}
            )
            transitions[(j, 0)].append(
                {"next_state": j + 1, "probability": 0.9, "reward": 0}
            )
        transitions[(length, 0)].append(
            {"next_state": length, "probability": 0.9, "reward": 1}
        )
        transitions[(length, 0)].append(
            {"next_state": length - 1, "probability": 0.1, "reward": 0}
        )

        return transitions


if __name__ == "__main__":
    from rllib.environment import GymEnvironment
    from rllib.environment.utilities import transitions2kernelreward

    import qreps  # noqa: F401

    env = GymEnvironment("RiverSwim-v0", length=5)
    kernel, reward = transitions2kernelreward(
        env.env.transitions, env.num_states, env.num_actions
    )
    print(kernel, reward)
    state = env.reset()
    print(state)
    for i in range(100):
        action = env.action_space.sample()
        next_state, r, done, f = env.step(action)
        print(state, action, next_state, r, done)
        state = next_state
