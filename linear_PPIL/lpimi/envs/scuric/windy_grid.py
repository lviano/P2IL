"""Python Script Template."""
from collections import defaultdict

import numpy as np
from rllib.environment.mdps import EasyGridWorld


class WindyGrid(EasyGridWorld):
    """Implementation of a LeftChain Chain Problem.

    Parameters
    ----------
    side: int
        number of states in chain.
    correct_prob: float
        probability of executing a correct transition.
    """

    def __init__(self, side=5, correct_prob=0.9):
        super().__init__(width=side, height=side, noise=1 - correct_prob)

    def _build_mdp(self, num_actions, noise, terminal_states=None):
        num_states = self.width * self.height
        transitions = defaultdict(list)
        for state in range(num_states):
            rew = 2 * num_states if state == 0 else 0
            grid_state = self._state_to_grid(state)  # (width, height)
            for action in range(num_actions):
                transitions[(state, action)].append(
                    {"next_state": state, "reward": rew, "probability": noise}
                )  # Noisy transitions
                grid_action = self._action_to_grid(action)
                grid_next_state = grid_state + grid_action
                if not self._is_valid(grid_next_state):
                    grid_next_state = grid_state
                next_state = self._grid_to_state(grid_next_state)
                transitions[(state, action)].append(
                    {"next_state": next_state, "reward": rew, "probability": 1 - noise}
                )
        return transitions


if __name__ == "__main__":
    from itertools import product

    from rllib.environment.utilities import transitions2kernelreward

    env = WindyGrid()
    kernel, reward = transitions2kernelreward(
        env.transitions, env.num_states, env.num_actions
    )
    for state, action in product(range(env.num_states), range(env.num_actions)):
        next_state = np.where(kernel[state, action])[0]
        np.where(kernel[state, action, next_state])
        print(
            env._state_to_grid(state),
            env._action_to_grid(action),
            env._state_to_grid(next_state),
            kernel[state, action, next_state],
            reward[state, action],
        )
