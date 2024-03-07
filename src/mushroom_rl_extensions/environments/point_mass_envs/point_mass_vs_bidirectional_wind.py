import numpy as np
from gymnasium import Env, spaces
from mushroom_rl.utils.viewer import Viewer
from mushroom_rl_extensions.core.environment import MDPInfo

from .abstract_point_mass_vs_wind import AbstractPointMassEnvironment


class PointMassBidirectionalWind(AbstractPointMassEnvironment):
    """
    Point mass environment where adversary can only push on x-axis
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_space_adv = spaces.Box(
            np.array([-self._new_adv_max_force], dtype=np.float32),
            np.array([self._new_adv_max_force], dtype=np.float32),
        )
        action_spaces = [self.action_space_prot, self.action_space_adv]
        mdp_info = MDPInfo(
            self.observation_space, action_spaces, self._gamma, self._horizon
        )
        self._mdp_info = mdp_info
        self.new_adv_max_force_magnitude = abs(self._new_adv_max_force)
