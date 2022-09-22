"""Python Script Template."""
from collections import defaultdict

from rllib.environment.mdp import MDP


class TwoState(MDP):
    """Implementation of a Two State Problem with stochastic transitions."""

    def __init__(self, reward_0=1.0):
        num_states = 2
        num_actions = 2
        transitions = self._build_mdp(reward_0=reward_0)
        self.check_transitions(transitions, num_states, num_actions)
        initial_state = 0
        super().__init__(transitions, num_states, num_actions, initial_state)

    @staticmethod
    def _build_mdp(reward_0):
        """Build the transition dictionary."""
        transitions = defaultdict(list)
        transitions[(0, 0)].append(
            {"next_state": 0, "probability": 1.0, "reward": reward_0}
        )
        transitions[(0, 1)].append({"next_state": 1, "probability": 1.0, "reward": 6.0})

        transitions[(1, 0)].append({"next_state": 0, "probability": 0.5, "reward": -3})
        transitions[(1, 0)].append({"next_state": 1, "probability": 0.5, "reward": -3})

        transitions[(1, 1)].append({"next_state": 0, "probability": 0.5, "reward": -3})
        transitions[(1, 1)].append({"next_state": 1, "probability": 0.5, "reward": -3})

        return transitions


if __name__ == "__main__":
    from itertools import product

    import torch
    from rllib.algorithms.tabular_planning import policy_iteration
    from rllib.environment.utilities import transitions2kernelreward

    gamma = 0.99
    env = TwoState(reward_0=1)
    kernel, reward = transitions2kernelreward(
        env.transitions, env.num_states, env.num_actions
    )
    policy, value = policy_iteration(env, gamma)
    print(
        torch.distributions.Categorical(logits=policy.table.detach()).probs,
        value.table.detach() * (1 - gamma),
    )
    for state, action in product(range(env.num_states), range(env.num_actions)):
        print(state, action, kernel[state, action], reward[state, action])
