import numpy as np

from dm_env import specs

from dm_control.utils import containers
from dm_control.rl import control
from dm_control.suite import common
from dm_control.suite.cartpole import Balance as CartPoleBalance
from dm_control.suite.cartpole import Physics as CartpolePhysics

_DEFAULT_TIME_LIMIT = 10
SUITE = containers.TaggedTasks()


def get_model_and_assets(num_poles=1):
    """Returns a tuple containing the model XML string and a dict of assets."""
    xml_string = common.read_model("cartpole.xml")
    return xml_string, common.ASSETS


@SUITE.add("benchmarking")
def balance_vs_adversary(
    time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None
):
    """Returns the Cartpole Balance task."""
    physics = PhysicsWithAdversary.from_xml_string(*get_model_and_assets())
    task = BalanceAgainstAdversary(swing_up=False, sparse=False, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs
    )


@SUITE.add("benchmarking")
def balance_sparse_vs_adversary(
    time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None
):
    """Returns the Cartpole Balance task with adversarial perturbation forces."""
    physics = PhysicsWithAdversary.from_xml_string(*get_model_and_assets())
    task = BalanceAgainstAdversary(swing_up=False, sparse=True, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs
    )


@SUITE.add("benchmarking")
def swingup_vs_adversary(
    time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None
):
    """Returns the Cartpole Swing-Up task."""
    physics = PhysicsWithAdversary.from_xml_string(*get_model_and_assets())
    task = BalanceAgainstAdversary(swing_up=True, sparse=False, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs
    )


@SUITE.add("benchmarking")
def swingup_sparse_vs_adversary(
    time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None
):
    """Returns the sparse reward variant of the Cartpole Swing-Up task."""
    physics = PhysicsWithAdversary.from_xml_string(*get_model_and_assets())
    task = BalanceAgainstAdversary(swing_up=True, sparse=True, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs
    )


class PhysicsWithAdversary(CartpolePhysics):
    """Physics simulation with adversarial forces."""

    def __init__(self, data):
        super().__init__(data)

        # Setup of adversary
        self._idx_perturbed_body = self.model.name2id("pole_1", "body")
        adv_max_force = 2.0
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
        Change the mass of pole_1
        """
        idx_torso = self.model.name2id("pole_1", "body")
        self.model.body_mass[idx_torso] = metric_value

    def change_second_metric(self, metric_value):
        """
        Change the mass of the cart
        """
        idx_torso = self.model.name2id("cart", "body")
        self.model.body_mass[idx_torso] = metric_value


class BalanceAgainstAdversary(CartPoleBalance):
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
