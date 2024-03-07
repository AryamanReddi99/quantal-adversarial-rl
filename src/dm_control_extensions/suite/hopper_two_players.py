import numpy as np

from dm_env import specs

from dm_control.rl import control
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards

from dm_control.suite.hopper import Hopper as HopperTask
from dm_control.suite.hopper import Physics as HopperPhysics


SUITE = containers.TaggedTasks()

_CONTROL_TIMESTEP = 0.02  # (Seconds)

# Default duration of an episode, in seconds.
_DEFAULT_TIME_LIMIT = 20

# Minimal height of torso over foot above which stand reward is 1.
_STAND_HEIGHT = 0.6

# Hopping speed above which hop reward is 1.
_HOP_SPEED = 2


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return common.read_model("hopper.xml"), common.ASSETS

@SUITE.add('benchmarking')
def stand_vs_adversary(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns a Hopper that strives to stand upright, balancing its pose."""
  physics = PhysicsWithAdversary.from_xml_string(*get_model_and_assets())
  task = HopperAgainstAdversary(hopping=False, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)

@SUITE.add("benchmarking")
def hop_vs_adversary(
    time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None
):
    """Returns a Hopper that strives to hop forward."""
    physics = PhysicsWithAdversary.from_xml_string(*get_model_and_assets())
    task = HopperAgainstAdversary(hopping=True, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs
    )



class PhysicsWithAdversary(HopperPhysics):
    """Physics simulation with adversarial forces."""

    def __init__(self, data):
        super().__init__(data)

        # Setup of adversary
        self._idx_perturbed_body = self.model.name2id("foot", "body")
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
        Change the torso mass
        """
        idx_torso = self.model.name2id("torso", "body")
        self.model.body_mass[idx_torso] = metric_value

    def change_second_metric(self, metric_value):
        """
        Change the tangential friction of all the geoms in the model
        """
        self.model.geom_friction[:, 0] = metric_value


class HopperAgainstAdversary(HopperTask):
    def get_reward(self, physics):
        standing = rewards.tolerance(physics.height(), (_STAND_HEIGHT, 2))
        if self._hopping:
            hopping = rewards.tolerance(
                physics.speed(),
                bounds=(_HOP_SPEED, float("inf")),
                margin=_HOP_SPEED / 2,
                value_at_margin=0.5,
                sigmoid="linear",
            )
            reward_protagonist = standing * hopping
        else:
            small_control = rewards.tolerance(physics.control(),
                                        margin=1, value_at_margin=0,
                                        sigmoid='quadratic').mean()
            small_control = (small_control + 4) / 5
            reward_protagonist = standing * small_control
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
