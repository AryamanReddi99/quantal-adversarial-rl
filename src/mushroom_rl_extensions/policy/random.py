import numpy as np
from mushroom_rl.policy import Policy


class RandomContinuousPolicy(Policy):
    def __init__(self, action_space):
        self.action_space = action_space

        self._add_save_attr(_approximator="mushroom!")

    def __call__(self, *args):
        raise NotImplementedError

    def draw_action(self, state):
        return self.action_space.sample()
