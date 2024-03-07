import numpy as np
from dm_control.rl import control
from dm_control.suite import common
from dm_control.suite.cheetah import Cheetah as CheetahTask
from dm_control.suite.cheetah import Physics as CheetahPhysics
from dm_control.utils import containers, rewards
from dm_env import specs

# How long the simulation will run, in seconds.
_DEFAULT_TIME_LIMIT = 10

# Running speed above which reward is 1.
_RUN_SPEED = 10

SUITE = containers.TaggedTasks()


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return common.read_model("cheetah.xml"), common.ASSETS


@SUITE.add("benchmarking")
def run_vs_adversary(
    time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None
):
    """Returns the run task with adversarial perturbation forces."""
    physics = PhysicsWithAdversary.from_xml_string(*get_model_and_assets())
    task = CheetahAgainstAdversary(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs
    )


class PhysicsWithAdversary(CheetahPhysics):
    """Physics simulation with adversarial forces."""

    def __init__(self, data):
        super().__init__(data)

        # Setup of adversary
        perturbed_body_names = ["torso", "bfoot", "ffoot"]
        self._idx_perturbed_bodies = [
            self.model.name2id(body_name, "body") for body_name in perturbed_body_names
        ]
        adv_max_force = 1.0
        adv_max_forces = np.ones(2 * len(self._idx_perturbed_bodies)) * adv_max_force
        adv_min_forces = -adv_max_forces
        self.adv_action_spec = specs.BoundedArray(
            adv_max_forces.shape,
            dtype=float,
            minimum=adv_min_forces,
            maximum=adv_max_forces,
        )

    def set_perturbation(self, adv_actions):
        """
        Applies perturbing forces of the adversary to the environment
        xfrc: 3D force and 3D torque in cartesian space applied at the center of mass of a certain body
        """
        # reset xfrc of previous time step
        new_xfrc = self.data.xfrc_applied * 0.0
        # clamp the provided actions to the adv action spec range
        adv_actions = np.clip(
            adv_actions, self.adv_action_spec.minimum, self.adv_action_spec.maximum
        )
        for i, idx_perturbed_body in enumerate(self._idx_perturbed_bodies):
            new_xfrc[idx_perturbed_body] = np.array(
                [adv_actions[i * 2], 0.0, adv_actions[i * 2 + 1], 0.0, 0.0, 0.0]
            )
        self.data.xfrc_applied[...] = new_xfrc

    def change_first_metric(self, metric_value):
        """
        Change the torso mass
        """
        idx_torso = self.model.name2id("torso", "body")
        self.model.body_mass[idx_torso] = metric_value

    def change_second_metric(self, metric_value):
        """
        Change the tangential fraction of all the geoms in the model
        """
        self.model.geom_friction[:, 0] = metric_value


class CheetahAgainstAdversary(CheetahTask):
    def get_reward(self, physics):
        reward_protagonist = rewards.tolerance(
            physics.speed(),
            bounds=(_RUN_SPEED, float("inf")),
            margin=_RUN_SPEED,
            value_at_margin=0,
            sigmoid="linear",
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
