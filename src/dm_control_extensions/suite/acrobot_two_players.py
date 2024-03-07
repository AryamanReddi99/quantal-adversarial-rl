import numpy as np

from dm_env import specs
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.suite.acrobot import Balance as AcrobotBalance
from dm_control.suite.acrobot import Physics as AcrobotPhysics

_DEFAULT_TIME_LIMIT = 10
SUITE = containers.TaggedTasks()


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return common.read_model("acrobot.xml"), common.ASSETS


@SUITE.add("benchmarking")
def swingup_vs_adversary(
    time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None
):
    """Returns Acrobot balance task."""
    physics = PhysicsWithAdversary.from_xml_string(*get_model_and_assets())
    task = BalanceAgainstAdversary(sparse=False, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs
    )


@SUITE.add("benchmarking")
def swingup_sparse_vs_adversary(
    time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None
):
    """Returns Acrobot balance task."""
    physics = PhysicsWithAdversary.from_xml_string(*get_model_and_assets())
    task = BalanceAgainstAdversary(sparse=True, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs
    )


class PhysicsWithAdversary(AcrobotPhysics):
    """Physics simulation with adversarial forces."""

    def __init__(self, data):
        super().__init__(data)

        # Setup of adversary
        self._idx_perturbed_body = self.model.name2id("upper_arm", "body")
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
        Change the mass of upper_arm
        """
        idx_upper_arm = self.model.name2id("upper_arm", "body")
        self.model.body_mass[idx_upper_arm] = metric_value

    def change_second_metric(self, metric_value):
        """
        Change the mass of lower arm
        """
        idx_lower_arm = self.model.name2id("lower_arm", "body")
        self.model.body_mass[idx_lower_arm] = metric_value


class BalanceAgainstAdversary(AcrobotBalance):
    def get_reward(self, physics):
        """Returns a sparse or a smooth reward, as specified in the constructor."""
        reward_protagonist = self._get_reward(physics, sparse=self._sparse)
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
