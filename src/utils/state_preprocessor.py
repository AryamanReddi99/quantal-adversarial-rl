import numpy as np

from mushroom_rl.core.serialization import Serializable


class StateVariableConcatenation(Serializable):
    def __init__(self):
        self.episode_count = 0
        self.variables = None

    def __call__(self, state):
        if self.variables is not None:
            current_variable = self.variables[self.episode_count]
            return np.concatenate((state, np.array([current_variable])), axis=0)
        else:
            assert False, "Preprocessor: No variables have been provided for training!"

    def set_variables(self, new_variables):
        self.variables = new_variables

    def update_episode_count(self, episode_count):
        self.episode_count = episode_count

    def reset_episode_count(self):
        self.episode_count = 0
