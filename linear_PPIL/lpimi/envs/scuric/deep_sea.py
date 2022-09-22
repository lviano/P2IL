"""Deep Sea implementation."""
from collections import defaultdict
from itertools import product

import numpy as np
from rllib.environment.mdps import EasyGridWorld


class DeepSea(EasyGridWorld):
    """Implementation of a DeepSea Chain Problem.

    Parameters
    ----------
    side: int
        number of states in chain.
    deterministic: bool
        probability of executing a correct transition.
    """

    def __init__(self, side=5, deterministic=False):
        action_mapping = np.random.randint(0, 2, side * side)
        self.action_mapping = np.stack((action_mapping, 1 - action_mapping), -1)
        self.deterministic = deterministic

        if self.deterministic:
            noise = 0
        else:
            noise = 1 / side

        super().__init__(width=side, height=side, num_actions=2, noise=noise)
        self.num_states = side * side

    def reset(self):
        """Reset MDP environment."""
        self._state = 0
        self._time = 0
        return self._state

    def reward(self, row, col, action):
        """Compute reward."""
        reward = 0.0

        if action == 1:  # Move right.
            reward -= 0.01 / self.width

        if (row == self.height - 1) and (col == self.width - 1):
            reward += 1 + 0.01 / self.width
            # else:
            # if not self.deterministic:
            # reward += np.random.randn()  # Noise rewards at the end of chain.

        return reward

    def _build_mdp(self, num_actions, noise, terminal_states=None):
        transitions = defaultdict(list)

        for (row, col) in product(range(self.height - 1), range(self.width)):
            state = self._grid_to_state(np.array([row, col]))

            for action in range(num_actions):
                direction = self.action_mapping[state, action]
                reward = self.reward(row, col, direction)
                next_col = col - 1 if direction == 0 else col + 1
                next_col = np.clip(next_col, 0, self.width - 1)

                next_state = self._grid_to_state(np.array([row + 1, next_col]))
                noisy_next_state = self._grid_to_state(np.array([row + 1, col]))
                transitions[(state, action)].append(
                    {
                        "next_state": noisy_next_state,
                        "reward": reward,
                        "probability": noise,
                    }
                )  # Noisy transitions
                transitions[(state, action)].append(
                    {
                        "next_state": next_state,
                        "reward": reward,
                        "probability": 1 - noise,
                    }
                )

        for col in range(self.width):
            for action in range(num_actions):
                row = self.height - 1
                state = self._grid_to_state(np.array([row, col]))
                direction = self.action_mapping[state, action]
                reward = self.reward(self.height - 1, col, direction)

                transitions[(state, action)].append(
                    {"next_state": 0, "reward": reward, "probability": 1}
                )  # Noisy transitions
        return transitions


if __name__ == "__main__":
    from rllib.environment import GymEnvironment
    from rllib.environment.utilities import transitions2kernelreward

    import qreps  # noqa: F401

    env = GymEnvironment("DeepSea-v0", side=5)
    kernel, reward = transitions2kernelreward(
        env.env.transitions, env.num_states, env.num_actions
    )
    state = env.reset()
    print(state)
    for i in range(10):
        action = env.action_space.sample()
        next_state, r, done, f = env.step(action)
        print(
            env.env._state_to_grid(state),
            env.env._state_to_grid(next_state),
            action,
            r,
            done,
        )
        state = next_state
