import numpy as np

from mushroom_rl.policy.policy import Policy


class ConstantPolicy(Policy):
    def __init__(self, action_space_shape, constant_value):
        self.action_space_shape = action_space_shape
        self.constant_value = constant_value

    def __call__(self, state, action):
        raise NotImplementedError

    def draw_action(self, state):
        sampled_action = np.ones(self.action_space_shape) * self.constant_value
        return sampled_action
