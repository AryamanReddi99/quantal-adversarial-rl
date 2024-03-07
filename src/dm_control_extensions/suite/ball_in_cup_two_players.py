import numpy as np

from dm_env import specs

from dm_control.rl import control
from dm_control.suite import common
from dm_control.utils import containers

from dm_control.suite.ball_in_cup import Physics as BallInCupPhysics
from dm_control.suite.ball_in_cup import BallInCup as BallInCupTask

_DEFAULT_TIME_LIMIT = 20  # (seconds)
_CONTROL_TIMESTEP = 0.02  # (seconds)

SUITE = containers.TaggedTasks()


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return common.read_model("ball_in_cup.xml"), common.ASSETS


@SUITE.add("benchmarking", "easy")
def catch_vs_adversary(
    time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None
):
    """Returns the Ball-in-Cup task."""
    physics = PhysicsWithAdversary.from_xml_string(*get_model_and_assets())
    task = BallInCupAgainstAdversary(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs
    )


class PhysicsWithAdversary(BallInCupPhysics):
    """Physics simulation with adversarial forces."""

    def __init__(self, data):
        super().__init__(data)

        # Setup of adversary
        self._idx_perturbed_body = self.model.name2id("ball", "body")
        adv_max_force = 0.005
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
        Change the mass of ball
        """
        idx_pole = self.model.name2id("ball", "body")
        self.model.body_mass[idx_pole] = metric_value

    def change_second_metric(self, metric_value):
        """
        Change the mass of cup
        """
        idx_mass = self.model.name2id("cup", "body")
        self.model.body_mass[idx_mass] = metric_value


class BallInCupAgainstAdversary(BallInCupTask):
    def get_reward(self, physics):
        protagonist_reward = physics.in_target()
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
