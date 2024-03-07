import numpy as np
from dm_control.rl import control
from dm_control.suite import common
from dm_control.suite.reacher import Physics as ReacherPhysics
from dm_control.suite.reacher import Reacher as ReacherTask
from dm_control.utils import containers, rewards
from dm_env import specs

SUITE = containers.TaggedTasks()
_DEFAULT_TIME_LIMIT = 20
_BIG_TARGET = 0.05
_SMALL_TARGET = 0.015


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return common.read_model("reacher.xml"), common.ASSETS


@SUITE.add("benchmarking", "easy")
def easy_vs_adversary(
    time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None
):
    """Returns reacher with sparse reward with 5e-2 tol and randomized target."""
    physics = PhysicsWithAdversary.from_xml_string(*get_model_and_assets())
    task = ReacherAgainstAdversary(target_size=_BIG_TARGET, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs
    )


@SUITE.add("benchmarking")
def hard_vs_adversary(
    time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None
):
    """Returns reacher with sparse reward with 1e-2 tol and randomized target."""
    physics = PhysicsWithAdversary.from_xml_string(*get_model_and_assets())
    task = ReacherAgainstAdversary(target_size=_SMALL_TARGET, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs
    )


class PhysicsWithAdversary(ReacherPhysics):
    """Physics simulation with adversarial forces."""

    def __init__(self, data):
        super().__init__(data)

        # Setup of adversary
        self._idx_perturbed_body = self.model.name2id("arm", "body")
        adv_max_force = 0.1
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
            [adv_action[0], adv_action[1], 0.0, 0.0, 0.0, 0.0]
        )
        self.data.xfrc_applied[...] = new_xfrc

    def change_first_metric(self, metric_value):
        """
        Change the mass of arm
        """
        idx_arm = self.model.name2id("arm", "body")
        self.model.body_mass[idx_arm] = metric_value

    def change_second_metric(self, metric_value):
        """
        Change the mass of hand
        """
        idx_hand = self.model.name2id("hand", "body")
        self.model.body_mass[idx_hand] = metric_value


class ReacherAgainstAdversary(ReacherTask):
    def get_reward(self, physics):
        radii = physics.named.model.geom_size[["target", "finger"], 0].sum()
        protagonist_reward = rewards.tolerance(
            physics.finger_to_target_dist(), (0, radii)
        )
        adversary_reward = -protagonist_reward

        return np.array([protagonist_reward, adversary_reward])

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
