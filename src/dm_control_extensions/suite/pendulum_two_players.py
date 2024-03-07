import os
import numpy as np

from dm_env import specs

from dm_control.utils import containers
from dm_control.rl import control
from dm_control.suite import common
from dm_control.utils import rewards
from dm_control.utils import io as resources
from dm_control.suite.pendulum import SwingUp as PendulumSwingUp
from dm_control.suite.pendulum import Physics as PendulumPhysics

_DEFAULT_TIME_LIMIT = 20
_ANGLE_BOUND = 8
_COSINE_BOUND = np.cos(np.deg2rad(_ANGLE_BOUND))
SUITE = containers.TaggedTasks()
_SUITE_DIR = os.path.dirname(__file__)

def read_model(model_filename):
  """Reads a model XML file and returns its contents as a string."""
  return resources.GetResource(os.path.join(_SUITE_DIR, model_filename))

def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return read_model("pendulum.xml"), common.ASSETS


@SUITE.add("benchmarking")
def swingup_vs_adversary(
    time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None
):
    """Returns pendulum swingup task ."""
    physics = PhysicsWithAdversary.from_xml_string(*get_model_and_assets())
    task = SwingUpAgainstAdversary(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs
    )


class PhysicsWithAdversary(PendulumPhysics):
    """Physics simulation with adversarial forces."""

    def __init__(self, data):
        super().__init__(data)

        # Setup of adversary
        self._idx_perturbed_body = self.model.name2id("pole", "body")
        adv_max_force = 1.0
        adv_max_forces = np.ones(2) * adv_max_force
        adv_min_forces = -adv_max_forces
        self.adv_action_spec = specs.BoundedArray(
            (2,), dtype=float, minimum=adv_min_forces, maximum=adv_max_forces
        )

    def set_perturbation(self, adv_action):
        """
        Applies perturbing forces of the adversary to the environment
        xfrc: 3D force and 3D torque in cartesian space applied at the center of mass of a certain body
        """
        # reset xfrc of previous time step
        new_xfrc = self.data.xfrc_applied * 0.0
        # clamp the provided actions to the adv action spec range
        adv_action = np.clip(
            adv_action, self.adv_action_spec.minimum, self.adv_action_spec.maximum
        )
        new_xfrc[self._idx_perturbed_body] = np.array(
            [adv_action[0], 0.0, adv_action[1], 0.0, 0.0, 0.0]
        )
        self.data.xfrc_applied[...] = new_xfrc

    def change_first_metric(self, metric_value):
        """
        Change the mass of pole
        """
        idx_pole = self.model.name2id("pole", "body")
        self.model.body_mass[idx_pole] = metric_value

    def change_second_metric(self, metric_value):
        """
        Change the mass of mass
        """
        idx_mass = self.model.name2id("mass", "body")
        self.model.body_mass[idx_mass] = metric_value


class SwingUpAgainstAdversary(PendulumSwingUp):
    def get_reward(self, physics):
        """Returns a sparse or a smooth reward, as specified in the constructor."""
        reward_protagonist = rewards.tolerance(
            physics.pole_vertical(), (_COSINE_BOUND, 1)
        )
        reward_adversary = -reward_protagonist
        return np.array([reward_protagonist, reward_adversary])

    def adv_action_spec(self, physics):
        return physics.adv_action_spec

    def update_adversary(self, physics, new_adv_max_force):
        adv_max_force = new_adv_max_force
        adv_action_spec_shape = physics.adv_action_spec.shape
        adv_max_forces = np.ones(adv_action_spec_shape) * adv_max_force
        adv_min_forces = -adv_max_forces
        physics.adv_action_spec = specs.BoundedArray(
            adv_action_spec_shape,
            dtype=float,
            minimum=adv_min_forces,
            maximum=adv_max_forces,
        )
