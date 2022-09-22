"""Python Script Template."""

import numpy as np
from gym import Wrapper


class RandomActionWrapper(Wrapper):
    """Wrapper that makes an environment random.

    With probability p, it executes an action uniformly at random.
    With probability 1-p, it executes the selected action.
    """

    def __init__(self, env, p=0):
        self.p = p
        super().__init__(env)

    def step(self, action):
        """Step with random action."""
        if np.random.rand() < self.p:
            action = self.action_space.sample()
        return super().step(action)
