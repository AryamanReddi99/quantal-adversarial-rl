import numpy as np
from gymnasium import Env, spaces
from mushroom_rl.utils.viewer import Viewer
from mushroom_rl_extensions.core.environment import MDPInfo


class AbstractPointMassEnvironment(Env):
    """
    General environment for point mass environment with wind adversary
    Abstract version of this env allows 2D forces from protagonist and adversary
    Env is a square centered around (0,0) with sides of length self.size
    Fully customisable, subclasses should implement a specific game
    [0, adv_wind_height] is the range within which the adversary can push
    prot_max_force is the maximum x and y force the protagonist can exert on itself
    new_adv_max_force is the maximum x and y force the adversary can exert on the protagonist
    """

    def __init__(
        self,
        horizon: int = 500,
        gamma: float = 0.99,
        bool_render: bool = False,
        size: float = 4.0,
        friction: float = 1.0,
        prot_mass: int = 5,
        start_state: np.ndarray = np.array([0.0, 0.0, -1.5, 0.0]),
        goal_state: np.ndarray = np.array([0.0, 0.0, 1.5, 0.0]),
        gap_x_pos: float = 0.0,
        gap_width: float = 1.0,
        adv_wind_height: float = -1.0,
        goal_tol: float = 0.1,
        prot_max_force: int = 1,
        new_adv_max_force: int = 10,
    ):
        self._horizon = horizon
        self._gamma = gamma
        self.bool_render = bool_render
        self._size = size
        self._friction = friction
        self._prot_mass = prot_mass
        self._state = None
        self._start_state = start_state
        self._goal_state = goal_state
        self._gap_x_pos = gap_x_pos
        self._gap_width = gap_width
        self._adv_wind_height = adv_wind_height
        self._goal_tol = goal_tol
        self._new_adv_max_force = new_adv_max_force

        self._dt = 0.01

        self.action_space_prot = spaces.Box(
            np.array([-prot_max_force, -prot_max_force], dtype=np.float32),
            np.array([prot_max_force, prot_max_force], dtype=np.float32),
        )
        self.action_space_adv = spaces.Box(
            np.array([-new_adv_max_force, -new_adv_max_force], dtype=np.float32),
            np.array([new_adv_max_force, new_adv_max_force], dtype=np.float32),
        )
        self.observation_space = spaces.Box(
            np.array([-self._size / 2, -np.inf, -self._size / 2, -np.inf]),
            np.array([self._size / 2, np.inf, self._size / 2, np.inf]),
        )

        action_spaces = [self.action_space_prot, self.action_space_adv]
        mdp_info = MDPInfo(
            self.observation_space, action_spaces, self._gamma, self._horizon
        )
        self._mdp_info = mdp_info

        # max magnitude force the adv can generate with vertical forces included
        self.new_adv_max_force_magnitude = np.linalg.norm(
            np.array(
                [
                    max(
                        abs(self.action_space_adv.low[0]),
                        abs(self.action_space_adv.high[0]),
                    ),
                    max(
                        abs(self.action_space_adv.low[1]),
                        abs(self.action_space_adv.high[1]),
                    ),
                ]
            )
        )

        self._viewer = Viewer(self._size, self._size, background=(255, 255, 255))

    def _get_info(self):
        distance = np.linalg.norm(self._goal_state[0::2] - self._state[0::2])
        return {
            "distance": distance,
            "success": distance <= self._goal_tol,
        }

    def _step_internal(self, state, action):
        state_der = np.zeros(4)  # state derivative
        state_der[0::2] = state[1::2]
        state_der[1::2] = (
            action - self._friction * state[1::2]
        ) / self._prot_mass  # F=ma

        new_state = state + self._dt * state_der

        # if prot reaches bounds of env, set speed in that direction to 0
        if (  # x
            new_state[0] < self.observation_space.low[0]
            or new_state[0] > self.observation_space.high[0]
        ):
            new_state[1] = 0
        if (  # y
            new_state[2] < self.observation_space.low[2]
            or new_state[2] > self.observation_space.high[2]
        ):
            new_state[3] = 0

        new_state = np.clip(
            new_state,
            self.observation_space.low,
            self.observation_space.high,
        )

        # passing y=0 line
        if state[2] >= 0 > new_state[2]:  # passing from above
            alpha = (0.0 - state[2]) / (new_state[2] - state[2])
            x_crit = (
                alpha * new_state[0] + (1 - alpha) * state[0]
            )  # interpolate the x-coord when prot passes y=0 line

            if (
                np.abs(x_crit - self._gap_x_pos) > 0.5 * self._gap_width
            ):  # prot stops at barrier
                new_state = np.array([x_crit, state[1], 1e-4, 0])

        if state[2] <= 0 < new_state[2]:  # passing from below
            alpha = (0.0 - state[2]) / (new_state[2] - state[2])
            x_crit = (
                alpha * new_state[0] + (1 - alpha) * state[0]
            )  # interpolate the x-coord when prot passes y=0 line

            if (
                np.abs(x_crit - self._gap_x_pos) > 0.5 * self._gap_width
            ):  # prot stops at barrier
                new_state = np.array([x_crit, state[1], -1e-4, 0])

        return new_state

    def reset(self, initial_state=None):
        if initial_state is None:
            initial_state = self._start_state
        self._state = initial_state

        # Starting adversary action
        self._adv_action = np.array([0, 0])

        return self._state

    def step(self, action):
        """
        accepts action list
        action[0] = protagonist action
        action[1] = adversary action
        """

        if self._state is None:
            raise RuntimeError(
                "State is None! Please call reset() on environment before taking a step."
            )

        prot_action = np.clip(
            action[0], self.action_space_prot.low, self.action_space_prot.high
        )
        adv_action = np.clip(
            action[1], self.action_space_adv.low, self.action_space_adv.high
        )

        net_action = prot_action

        if self._adv_wind_height <= self._state[2] <= 0:  # prot is in adv's wind region
            if adv_action.shape == (1,):  # only horizontal force
                adv_action = np.append(adv_action, 0)  # make horizontal force 2D
            net_action += adv_action

        self._adv_action = adv_action  # store this so it can be rendered

        new_state = self._state
        for i in range(0, 10):
            new_state = self._step_internal(new_state, net_action)

        self._state = new_state
        info = self._get_info()
        absorbing = info["success"]
        # reward = 0 if info["success"]
        if absorbing:
            reward = 0
        else:
            reward = np.exp(-1.0 * info["distance"]) - 1
            # reward = np.exp(-1.0 * info["distance"])

        # print(adv_action)

        # PLOTTING DEBUG
        # absorbing = False

        return new_state, [reward, -reward], absorbing, info

    def render(self, render_info):
        pos = self._gap_x_pos + self._size / 2  # center of gap
        self._viewer.line(  # left wall
            np.array([0, self._size / 2]),
            np.array(
                [np.clip(pos - 0.5 * self._gap_width, 0.0, self._size), self._size / 2]
            ),
            color=(0, 0, 0),
            width=5,
        )

        self._viewer.line(  # right wall
            np.array(
                [
                    np.clip(
                        pos + 0.5 * self._gap_width,
                        0.0,
                        self._size,
                    ),
                    self._size / 2,
                ]
            ),
            np.array([self._size, self._size / 2]),
            color=(0, 0, 0),
            width=5,
        )

        self._viewer.circle(  # protagonist
            self._state[0::2] + np.array([self._size / 2, self._size / 2]),
            0.1,
            color=(0, 0, 250),
        )

        self._viewer.square(  # goal
            self._goal_state[0::2] + np.array([self._size / 2, self._size / 2]),
            angle=0,
            edge=self._goal_tol * 2,
            color=(0, 200, 0),
        )

        # adversary wind arrow
        if self._adv_action.shape == (1,):
            direction = np.array([self._adv_action[0], 0])
        else:
            direction = self._adv_action
        self._viewer.force_arrow(
            center=np.array(
                [self._size / 2, self._size / 2 + self._adv_wind_height / 2]
            ),
            direction=direction,
            force=np.linalg.norm(self._adv_action),
            max_force=self.new_adv_max_force_magnitude,
            max_length=self._size / 4,
            color=(200, 0, 0),
            width=4,
        )

        # the region the adversary affects
        self._viewer.line(
            start=np.array([0, self._size / 2 + self._adv_wind_height]),
            end=np.array([self._size, self._size / 2 + self._adv_wind_height]),
            color=(200, 0, 0),
            width=1,
        )

        self._viewer.display(self._dt)

    def stop(self):
        if self.bool_render:
            self._viewer.close()

    @property
    def info(self):
        """
        Returns:
             An object containing the info of the environment.

        """
        return self._mdp_info
