import warnings

import mujoco as mj

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from dm_control_extensions import suite

    # from dm_control import suite
    from dm_control.suite.wrappers import pixels

from mushroom_rl.utils.spaces import *
from mushroom_rl_extensions.core.environment import Environment, MDPInfo


class DMControl(Environment):
    """
    Interface for dm_control suite Mujoco environments. It makes it possible to
    use every dm_control suite Mujoco environment just providing the necessary
    information.

    """

    def __init__(
        self,
        domain_name,
        task_name,
        horizon=None,
        gamma=0.99,
        task_kwargs=None,
        dt=0.01,
        width_screen=800,
        height_screen=800,
        camera_id=0,
        use_pixels=False,
        pixels_width=64,
        pixels_height=64,
    ):
        """
        Constructor.

        Args:
             domain_name (str): name of the environment;
             task_name (str): name of the task of the environment;
             horizon (int): the horizon;
             gamma (float): the discount factor;
             task_kwargs (dict, None): parameters of the task;
             dt (float, .01): duration of a control step;
             width_screen (int, 480): width of the screen;
             height_screen (int, 480): height of the screen;
             camera_id (int, 0): position of camera to render the environment;
             use_pixels (bool, False): if True, pixel observations are used
                rather than the state vector;
             pixels_width (int, 64): width of the pixel observation;
             pixels_height (int, 64): height of the pixel observation;

        """
        # MDP creation
        self.env = suite.load(
            domain_name, task_name, task_kwargs=task_kwargs, visualize_reward=False
        )
        if use_pixels:
            self.env = pixels.Wrapper(
                self.env, render_kwargs={"width": pixels_width, "height": pixels_height}
            )

        # get the default horizon
        if horizon is None:
            horizon = self.env._step_limit

        # Hack to ignore dm_control time limit.
        self.env._step_limit = np.inf

        if use_pixels:
            self._convert_observation_space = self._convert_observation_space_pixels
            self._convert_observation = self._convert_observation_pixels
        else:
            self._convert_observation_space = self._convert_observation_space_vector
            self._convert_observation = self._convert_observation_vector

        # MDP properties
        action_spec_prot = self.env.action_spec()
        try:
            action_spec_adv = self.env.task.adv_action_spec(self.env.physics)
        except:
            action_spec_adv = None
        action_space = self._convert_action_space(action_spec_prot, action_spec_adv)
        observation_space = self._convert_observation_space(self.env.observation_spec())
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        super().__init__(mdp_info)

        self._state = None

        # Rendering
        self._domain_name = domain_name
        self._task_name = task_name
        self._dt = dt
        self._width_screen = width_screen
        self._height_screen = height_screen
        self._camera_id = camera_id
        self.render_init = False

        if "quadruped" in self._domain_name:
            self._camera_id_zoom = 1

    def reset(self, state=None):
        if state is None:
            self._state = self._convert_observation(self.env.reset().observation)
        else:
            raise NotImplementedError

        return self._state

    def step(self, action):
        if len(action) == 2:
            self.env.physics.set_perturbation(action[1])

        control_dtype = self.env.physics.data.ctrl.dtype
        actuator_action = action[0].astype(control_dtype)

        step = self.env.step(actuator_action)

        reward = step.reward
        self._state = self._convert_observation(step.observation)
        absorbing = step.last()

        if "Quadruped" in type(self.env.task).__name__:
            # TERMINATE IF FALL OVER
            if self.env.physics.torso_upright() <= 0:
                absorbing = True
                # pass

        return self._state, reward, absorbing, {}

    def render(self, render_info):
        if not self.render_init:
            self._cam = mj.MjvCamera()
            self._cam.fixedcamid = self._camera_id
            self._cam.type = mj.mjtCamera.mjCAMERA_FIXED
            self._opt = mj.MjvOption()
            mj.glfw.glfw.init()
            self._window = mj.glfw.glfw.create_window(
                self._width_screen, self._height_screen, "Mujoco Viewer", None, None
            )
            self._viewport = mj.MjrRect(0, 0, self._width_screen, self._height_screen)
            mj.glfw.glfw.make_context_current(self._window)
            mj.glfw.glfw.swap_interval(1)
            mj.mjv_defaultOption(self._opt)
            self._scene = mj.MjvScene(model=self.env.physics.model.ptr, maxgeom=10000)
            self._context = mj.MjrContext(
                self.env.physics.model.ptr, mj.mjtFontScale.mjFONTSCALE_150.value
            )
            self._perturb = mj.MjvPerturb()
            self._perturb.active = 0
            self._perturb.select = 0
            self.render_init = True

        self._opt.flags[mj.mjtVisFlag.mjVIS_PERTFORCE] = True
        mj.mjv_updateScene(
            self.env.physics.model.ptr,
            self.env.physics.data.ptr,
            self._opt,
            self._perturb,
            self._cam,
            mj.mjtCatBit.mjCAT_ALL,
            self._scene,
        )
        mj.mjr_render(self._viewport, self._scene, self._context)

        ## Render Info
        # Domain Task Text
        msg_domain = f"Domain name: {self._domain_name}"
        msg_task = f"Task name:     {self._task_name}"
        mj.mjr_text(
            font=200,
            txt=msg_domain,
            con=self._context,
            x=0.05,
            y=0.9,
            r=0,
            g=1,
            b=0,
        )
        mj.mjr_text(
            font=200,
            txt=msg_task,
            con=self._context,
            x=0.05,
            y=0.85,
            r=0,
            g=1,
            b=0,
        )

        # Actions
        try:
            action = render_info["action"]
            # Action Text
            msg_prot = f"Protagonist action: {[round(el,2) for el in action[0]]}"
            mj.mjr_text(
                font=200,
                txt=msg_prot,
                con=self._context,
                x=0.05,
                y=0.1,
                r=1,
                g=1,
                b=0,
            )
            if len(action) == 2:
                msg_adv = f"Adversary action: {[round(el,2) for el in action[1]]}"
                mj.mjr_text(
                    font=200,
                    txt=msg_adv,
                    con=self._context,
                    x=0.05,
                    y=0.05,
                    r=1,
                    g=1,
                    b=0,
                )
        except:
            pass

        # Reward
        try:
            reward = render_info["reward"][0]
            # Action Text
            msg_reward = f"Reward: {reward}"
            mj.mjr_text(
                font=200,
                txt=msg_reward,
                con=self._context,
                x=0.05,
                y=0,
                r=1,
                g=1,
                b=0,
            )
        except:
            pass

        mj.glfw.glfw.swap_buffers(self._window)
        mj.glfw.glfw.poll_events()

    def stop(self):
        mj.glfw.glfw.terminate()
        self.render_init = False
        pass

    @staticmethod
    def _convert_observation_space_vector(observation_space):
        observation_shape = 0
        for i in observation_space:
            shape = observation_space[i].shape
            observation_var = 1
            for dim in shape:
                observation_var *= dim
            observation_shape += observation_var

        return Box(low=-np.inf, high=np.inf, shape=(observation_shape,))

    @staticmethod
    def _convert_observation_space_pixels(observation_space):
        img_size = observation_space["pixels"].shape
        return Box(low=0.0, high=255.0, shape=(3, img_size[0], img_size[1]))

    @staticmethod
    def _convert_action_space(action_spec_prot, action_spec_adv):
        # protagonist
        low_prot = action_spec_prot.minimum
        high_prot = action_spec_prot.maximum
        action_space_prot = Box(low=np.array(low_prot), high=np.array(high_prot))

        # adversary
        if action_spec_adv:
            low_adv = action_spec_adv.minimum
            high_adv = action_spec_adv.maximum
            action_space_adv = Box(low=np.array(low_adv), high=np.array(high_adv))
            return [action_space_prot, action_space_adv]

        return [action_space_prot]

    @staticmethod
    def _convert_observation_vector(observation):
        obs = list()
        for i in observation:
            obs.append(np.atleast_1d(observation[i]).flatten())

        return np.concatenate(obs)

    @staticmethod
    def _convert_observation_pixels(observation):
        return observation["pixels"].transpose((2, 0, 1))
