import collections
import os

import numpy as np
from dm_control.rl import control
from dm_control.suite import common
from dm_control.suite.quadruped import Move as Quadruped
from dm_control.suite.quadruped import Physics as QuadrupedPhysics
from dm_control.utils import containers
from dm_control.utils import io as resources
from dm_control.utils import rewards, xml_tools
from dm_env import specs
from lxml import etree
from scipy import ndimage

_DEFAULT_TIME_LIMIT = 20
_CONTROL_TIMESTEP = 0.02

# Horizontal speeds above which the move reward is 1.
_RUN_SPEED = 5
_WALK_SPEED = 0.5

# Constants related to terrain generation.
_HEIGHTFIELD_ID = 0
_TERRAIN_SMOOTHNESS = 0.15  # 0.0: maximally bumpy; 1.0: completely smooth.
_TERRAIN_BUMP_SCALE = 2  # Spatial scale of terrain bumps (in meters).

# Named model elements.
_TOES = ["toe_front_left", "toe_back_left", "toe_back_right", "toe_front_right"]
_WALLS = ["wall_px", "wall_py", "wall_nx", "wall_ny"]
_QUAD = [
    "torso",
    "hip_front_left",
    "knee_front_left",
    "ankle_front_left",
    "toe_front_left",
    "hip_front_right",
    "knee_front_right",
    "ankle_front_right",
    "toe_front_right",
    "hip_back_left",
    "knee_back_left",
    "ankle_back_left",
    "toe_back_left",
    "hip_back_right",
    "knee_back_right",
    "ankle_back_right",
    "toe_back_right",
]

SUITE = containers.TaggedTasks()
_SUITE_DIR = os.path.dirname(__file__)


def read_model(model_filename):
    """Reads a model XML file and returns its contents as a string."""
    return resources.GetResource(os.path.join(_SUITE_DIR, model_filename))


def make_model(
    xml, floor_size=None, terrain=False, rangefinders=False, walls_and_ball=False
):
    """Returns the model XML string."""
    try:
        xml_string = read_model(xml)
    except:
        xml_string = common.read_model("quadruped.xml")
    parser = etree.XMLParser(remove_blank_text=True)
    mjcf = etree.XML(xml_string, parser)

    # Set floor size.
    if floor_size is not None:
        floor_geom = mjcf.find(".//geom[@name='floor']")
        floor_geom.attrib["size"] = f"{floor_size} {floor_size} .5"

    # Remove walls, ball and target.
    if not walls_and_ball:
        for wall in _WALLS:
            wall_geom = xml_tools.find_element(mjcf, "geom", wall)
            wall_geom.getparent().remove(wall_geom)

        # Remove ball.
        ball_body = xml_tools.find_element(mjcf, "body", "ball")
        ball_body.getparent().remove(ball_body)

        # Remove target.
        target_site = xml_tools.find_element(mjcf, "site", "target")
        target_site.getparent().remove(target_site)

    # Remove terrain.
    if not terrain:
        terrain_geom = xml_tools.find_element(mjcf, "geom", "terrain")
        terrain_geom.getparent().remove(terrain_geom)

    # Remove rangefinders if they're not used, as range computations can be
    # expensive, especially in a scene with heightfields.
    if not rangefinders:
        rangefinder_sensors = mjcf.findall(".//rangefinder")
        for rf in rangefinder_sensors:
            rf.getparent().remove(rf)

    return etree.tostring(mjcf, pretty_print=True)


@SUITE.add("benchmarking")
def walk_vs_adversary(
    time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None
):
    """Returns the Walk task."""
    xml_string = make_model("quadruped.xml")
    physics = PhysicsWithAdversary.from_xml_string(xml_string, common.ASSETS)
    task = QuadrupedMoveAgainstAdversary(desired_speed=_WALK_SPEED, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs,
    )


@SUITE.add("benchmarking")
def run_vs_adversary(
    time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None
):
    """Returns the Walk task."""
    xml_string = make_model("quadruped.xml")
    physics = PhysicsWithAdversary.from_xml_string(xml_string, common.ASSETS)
    task = QuadrupedMoveAgainstAdversary(desired_speed=_RUN_SPEED, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs,
    )


@SUITE.add("benchmarking")
def reach_goal_vs_wind_adversary(
    time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None
):
    """Returns the reach goal task."""
    xml_string = make_model("quadruped_goal.xml")
    physics = PhysicsWithWindAdversary.from_xml_string(xml_string, common.ASSETS)
    task = QuadrupedGoalAgainstWindAdversary(desired_speed=_WALK_SPEED, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs,
    )


@SUITE.add("benchmarking")
def reach_goal_vs_adversary(
    time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None
):
    """Returns the reach goal task."""
    xml_string = make_model("quadruped_goal.xml")
    physics = PhysicsWithAdversary.from_xml_string(xml_string, common.ASSETS)
    task = QuadrupedGoalAgainstAdversary(desired_speed=_WALK_SPEED, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs,
    )


class PhysicsWithAdversary(QuadrupedPhysics):
    """Physics simulation with adversarial forces."""

    def __init__(self, data):
        super().__init__(data)

        # Setup of adversary
        perturbed_body_names = [
            "torso",
        ]
        self._idx_perturbed_bodies = [
            self.model.name2id(body_name, "body") for body_name in perturbed_body_names
        ]
        adv_max_force = 10.0
        adv_max_forces = np.ones(3 * len(self._idx_perturbed_bodies)) * adv_max_force
        adv_min_forces = -adv_max_forces
        self.adv_action_spec = specs.BoundedArray(
            adv_max_forces.shape,
            dtype=float,
            minimum=adv_min_forces,
            maximum=adv_max_forces,
        )

    def torso_x_velocity(self):
        """Returns the velocity of the torso, in the global frame."""
        return self.named.data.sensordata["torso_subtreelinvel"][0]

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
                [
                    adv_actions[i * 3],
                    0.0,
                    adv_actions[i * 3 + 1],
                    0.0,
                    adv_actions[i * 3 + 2],
                    0.0,
                ]
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


class PhysicsWithWindAdversary(QuadrupedPhysics):
    """Physics simulation with wind adversarial forces."""

    def __init__(self, data):
        super().__init__(data)

        # Setup of adversary
        perturbed_body_names = _QUAD
        self._idx_perturbed_bodies = [
            self.model.name2id(body_name, "body") for body_name in perturbed_body_names
        ]
        self.adv_max_force = 600
        adv_max_forces = np.ones(1) * self.adv_max_force
        adv_min_forces = 0 * adv_max_forces
        self.adv_action_spec = specs.BoundedArray(
            adv_max_forces.shape,
            dtype=float,
            minimum=adv_min_forces,
            maximum=adv_max_forces,
        )

    def torso_x_velocity(self):
        """Returns the velocity of the torso, in the global frame."""
        return self.named.data.sensordata["torso_subtreelinvel"][0]

    def torso_to_goal(self):
        """Returns the vector from torso to goal in global coordinates."""
        return (
            self.named.data.geom_xpos["goal", :2]
            - self.named.data.geom_xpos["torso", :2]
        )

    def torso_to_goal_dist(self):
        """Returns the signed distance between the torso and target."""
        return np.linalg.norm(self.torso_to_goal())

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
        unit_force = adv_actions[0] / len(self._idx_perturbed_bodies)
        for idx_perturbed_body in self._idx_perturbed_bodies:
            new_xfrc[idx_perturbed_body] = np.array(
                [0.0, unit_force, 0.0, 0.0, 0.0, 0.0]
            )

            # XFRC DEBUG
            # new_xfrc[idx_perturbed_body] = np.array([0, 500.0, 0.0, 0.0, 0.0, 0.0])

        # Adversary only affects from 4 <= x <= 8
        torso_x = self.named.data.geom_xpos["torso", :1][0]
        if 4 <= torso_x <= 8:
            self.data.xfrc_applied[...] = new_xfrc
        else:
            self.data.xfrc_applied[...] = new_xfrc * 0.0

    def change_mass(self, new_mass):
        idx_torso = self.model.name2id("torso", "body")
        self.model.body_mass[idx_torso] = new_mass

    def change_friction(self, new_friction):
        """
        Change the tangential fraction of all the geoms in the model
        """
        self.model.geom_friction[:, 0] = new_friction


def _common_observations(physics):
    """Returns the observations common to all tasks."""
    obs = collections.OrderedDict()
    obs["egocentric_state"] = physics.egocentric_state()
    obs["torso_velocity"] = physics.torso_velocity()
    obs["torso_upright"] = physics.torso_upright()
    obs["imu"] = physics.imu()
    obs["force_torque"] = physics.force_torque()
    return obs


def _upright_reward(physics, deviation_angle=0):
    """Returns a reward proportional to how upright the torso is.

    Args:
      physics: an instance of `Physics`.
      deviation_angle: A float, in degrees. The reward is 0 when the torso is
        exactly upside-down and 1 when the torso's z-axis is less than
        `deviation_angle` away from the global z-axis.
    """
    deviation = np.cos(np.deg2rad(deviation_angle))
    return rewards.tolerance(
        physics.torso_upright(),
        bounds=(deviation, float("inf")),
        sigmoid="linear",
        margin=1 + deviation,
        value_at_margin=0,
    )


class QuadrupedMoveAgainstAdversary(Quadruped):
    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.

        Args:
        physics: An instance of `Physics`.

        """
        # Initial configuration.
        super(Quadruped, self).initialize_episode(physics)

    def get_reward(self, physics):
        """Returns a reward to the agent."""

        ## Move reward term.
        move_reward = rewards.tolerance(
            physics.torso_velocity()[0],
            bounds=(self._desired_speed, float("inf")),
            margin=self._desired_speed,
            value_at_margin=0.5,
            sigmoid="linear",
        )
        reward_protagonist = _upright_reward(physics) * move_reward
        # reward_protagonist = move_reward
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


class QuadrupedGoalAgainstAdversary(Quadruped):
    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.

        Args:
        physics: An instance of `Physics`.

        """
        # Initial configuration.
        super(Quadruped, self).initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation to the agent."""
        obs = _common_observations(physics)
        obs["to_goal"] = physics.torso_to_goal()
        return obs

    def get_reward(self, physics):
        """Returns a reward to the agent."""

        # Move reward term.
        # reward_protagonist = 5 * np.exp(-0.2 * (physics.torso_to_goal_dist()))
        reward_protagonist = (
            _upright_reward(physics) * 5 * np.exp(-0.2 * (physics.torso_to_goal_dist()))
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


class QuadrupedGoalAgainstWindAdversary(Quadruped):
    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.

        Args:
        physics: An instance of `Physics`.

        """
        # Initial configuration.
        super(Quadruped, self).initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation to the agent."""
        obs = _common_observations(physics)
        obs["to_goal"] = physics.torso_to_goal()
        return obs

    def get_reward(self, physics):
        """Returns a reward to the agent."""

        # Move reward term.
        # reward_protagonist = 5 * np.exp(-0.2 * (physics.torso_to_goal_dist()))
        reward_protagonist = (
            _upright_reward(physics) * 5 * np.exp(-0.2 * (physics.torso_to_goal_dist()))
        )
        reward_adversary = -reward_protagonist
        return np.array([reward_protagonist, reward_adversary])

    def adv_action_spec(self, physics):
        return physics.adv_action_spec

    def update_adversary(self, physics, new_adv_max_force):
        adv_max_force = new_adv_max_force
        adv_action_spec_shape = physics.adv_action_spec.shape
        adv_max_forces = np.ones(adv_action_spec_shape) * adv_max_force
        adv_min_forces = 0 * adv_max_forces
        physics.adv_action_spec = specs.BoundedArray(
            adv_action_spec_shape,
            dtype=float,
            minimum=adv_min_forces,
            maximum=adv_max_forces,
        )
