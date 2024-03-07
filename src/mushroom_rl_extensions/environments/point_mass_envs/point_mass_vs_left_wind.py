import numpy as np
from gymnasium import spaces
from mushroom_rl_extensions.core.environment import MDPInfo

from .abstract_point_mass_vs_wind import AbstractPointMassEnvironment


class PointMassLeftWind(AbstractPointMassEnvironment):
    """
    Point mass environment where adversary can only push left
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self._new_adv_max_force < 0:
            self.action_space_adv = spaces.Box(
                np.array([self._new_adv_max_force], dtype=np.float32),
                np.array([0], dtype=np.float32),
            )
        else:
            self.action_space_adv = spaces.Box(
                np.array([-self._new_adv_max_force], dtype=np.float32),
                np.array([0], dtype=np.float32),
            )
        action_spaces = [self.action_space_prot, self.action_space_adv]
        mdp_info = MDPInfo(
            self.observation_space, action_spaces, self._gamma, self._horizon
        )
        self._mdp_info = mdp_info
        self.new_adv_max_force_magnitude = abs(self._new_adv_max_force)

    def reset(self, initial_state=None):
        if initial_state is None:
            initial_state = self._start_state
        self._state = initial_state

        # Starting adversary action
        self._adv_action = np.array([0])

        return self._state
