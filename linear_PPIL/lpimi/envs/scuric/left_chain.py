"""Python Script Template."""
from collections import defaultdict

from rllib.environment.mdp import MDP


class LeftChain(MDP):
    """Implementation of a LeftChain Chain Problem.

    Parameters
    ----------
    chain_length: int
        number of states in chain.
    correct_prob: float
        probability of executing a correct transition.
    """

    def __init__(self, chain_length=5, correct_prob=0.9):
        num_states = chain_length
        num_actions = 2
        transitions = self._build_mdp(num_states, correct_prob=correct_prob)
        self.check_transitions(transitions, num_states, num_actions)
        initial_state = 0
        super().__init__(transitions, num_states, num_actions, initial_state)

    @staticmethod
    def _build_mdp(chain_length, correct_prob):
        """Build the transition dictionary."""
        transitions = defaultdict(list)
        num_states = chain_length
        RIGHT = 1
        LEFT = 0

        for state in range(1, num_states - 1):
            for action in [LEFT, RIGHT]:
                next_state = state + 1 if action == RIGHT else state - 1
                transitions[(state, action)].append(
                    {"next_state": next_state, "probability": correct_prob, "reward": 0}
                )
                transitions[(state, action)].append(
                    {"next_state": state, "probability": 1 - correct_prob, "reward": 0}
                )
        # Initial transition.
        for action in [RIGHT, LEFT]:
            state, next_state = 0, num_states - 1
            rew = 0 if action == RIGHT else num_states
            transitions[(state, action)].append(
                {"next_state": next_state, "probability": correct_prob, "reward": rew}
            )
            transitions[(state, action)].append(
                {"next_state": state, "probability": 1 - correct_prob, "reward": 0}
            )
        # Final transition.
        for action in [RIGHT, LEFT]:
            state, next_state = num_states - 1, num_states - 2
            transitions[(state, action)].append(
                {"next_state": next_state, "probability": correct_prob, "reward": 0}
            )
            transitions[(state, action)].append(
                {
                    "next_state": state,
                    "probability": 1 - correct_prob,
                    "reward": 0,
                }
            )

        return transitions


if __name__ == "__main__":
    from itertools import product

    from rllib.algorithms.tabular_planning import policy_iteration
    from rllib.environment.utilities import transitions2kernelreward

    env = LeftChain()
    kernel, reward = transitions2kernelreward(
        env.transitions, env.num_states, env.num_actions
    )
    print(kernel)
    print(reward)
    policy, value = policy_iteration(env, 0.99)
    print(policy.table, value.table)
    for state, action in product(range(env.num_states), range(env.num_actions)):
        print(state, action, kernel[state, action], reward[state, action])
