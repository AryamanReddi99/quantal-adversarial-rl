from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from mushroom_rl.core.logger.logger import Logger
from mushroom_rl.core.serialization import Serializable
from mushroom_rl.utils.callbacks import CollectDataset
from mushroom_rl.utils.spaces import Box
from mushroom_rl_extensions.agents.create_agent import SetupAgent
from mushroom_rl_extensions.core.mas_core import MASCore
from mushroom_rl_extensions.core.multi_agent_core import MultiAgentCore
from mushroom_rl_extensions.core.multi_player_agent import MultiPlayerAgent
from mushroom_rl_extensions.core.self_paced_force_core import SelfPacedForceCore
from mushroom_rl_extensions.core.self_paced_rl_core import SelfPacedRLCore
from mushroom_rl_extensions.environments.mujoco_envs import dm_control_env
from mushroom_rl_extensions.policy.constant import ConstantPolicy
from mushroom_rl_extensions.utils.dataset import (
    compute_J,
    compute_quadruped_success_rate,
)
from tqdm import trange


class AbstractExperiment(ABC):
    """
    Abstract MuJoCo experiment class. Extend to create specific experiment type.

    Each domain has its own set of robustness values it can be tested over. Generally, these robustness values will dictate
    the mass of one of the bodies of the agent, and/or the tangential friction coefficient of the mdp. The specific details
    of what these metrics are domain-specific and can be found in the files in src/dm_control_extensions/<domain_name>.py

    domain_name, task_name: domain and task name associated with two-player mdp
    horizon: mdp horizon
    gamma: mdp discount factor
    bool_render: choose whether to render environment to glfw-compatible monitor
    new_adv_max_force: maximum force the adversary can use
    agent: the type of algorithm agent to use (results use SAC)
    results_dir: location of logging and file outputs
    seed: the seed of the current experiment run
    use_cuda: bool to enable cuda agents
    n_total_iterations: how many iterations to run (training loops over both agents)
    n_iterations_per_agent: how many internal iterations to run per agent (keep 1 for efficiency)
    n_steps_per_iteration: how many steps to train each agent for 1 iteration (do not set if using n_episodes_per_iteration)
    n_episodes_per_iteration: how many episodes to train each agent for 1 iteration (do not set if using n_steps_per_iteration)
    n_steps_per_fit: how many steps to wait before calling fit() on each agent
    n_evaluation_episodes: how many rollouts to evaluate performance
    n_evaluation_episodes_per_metric_value: how many rollouts to evaluate performance for a particular set of robustness metric values (keep 1 for efficiency)
    """

    def __init__(
        self,
        domain_name: str = None,
        task_name: str = None,
        horizon: int = None,
        gamma: float = 0.99,
        bool_render: bool = False,
        new_adv_max_force: float = None,
        agent: str = "sac",
        results_dir: str = "",
        seed: str = "0",
        use_cuda: bool = False,
        n_total_iterations: int = 200,
        n_total_iterations_worst_adversary: int = 0,
        n_iterations_per_agent: int = 1,
        n_steps_per_iteration: int = None,
        n_episodes_per_iteration: int = 5,
        n_steps_per_fit: int = 1,  # gets overwritten
        n_evaluation_episodes: int = 10,
        n_evaluation_episodes_per_metric_value: int = 1,
    ):
        self._domain_name = domain_name
        self._task_name = task_name
        self._horizon = horizon
        self._gamma = gamma
        self.bool_render = bool_render
        self.new_adv_max_force = new_adv_max_force
        self.agent = agent
        self.use_cuda = use_cuda
        self.results_dir = results_dir
        self.seed = seed
        self.n_total_iterations = n_total_iterations
        self.n_total_iterations_worst_adversary = n_total_iterations_worst_adversary
        self.n_iterations_per_agent = n_iterations_per_agent
        self.n_steps_per_iteration = n_steps_per_iteration
        self.n_episodes_per_iteration = n_episodes_per_iteration
        self.n_steps_per_fit = n_steps_per_fit
        self.n_evaluation_episodes = n_evaluation_episodes
        self.n_evaluation_episodes_per_metric_value = (
            n_evaluation_episodes_per_metric_value
        )

        self.environment_params = {
            "domain_name": self._domain_name,
            "task_name": self._task_name,
            "horizon": self._horizon,
            "gamma": self._gamma,
        }

        self.exp_logger = Logger(
            log_name="Experiment",
            results_dir=Path(self.results_dir) / "Logging",
            log_console=True,
            seed=self.seed,
            console_log_level=30,
        )
        self.adjust_steps_per_fit(agent)
        self.log_params()

        # Range of robustness values to evaluate
        if self._domain_name == "acrobot_two_players":
            first_metric_range = np.linspace(0.95, 1.05, 11)
            second_metric_range = np.linspace(0.95, 1.05, 11)
        elif self._domain_name == "ball_in_cup_two_players":
            first_metric_range = np.linspace(0.01, 0.11, 11)
            second_metric_range = np.linspace(0.01, 0.11, 11)
        if self._domain_name == "cartpole_two_players":
            if self._task_name in (
                "balance_vs_adversary",
                "balance_sparse_vs_adversary",
            ):
                first_metric_range = np.linspace(1, 20, 11)
                second_metric_range = np.linspace(0.5, 1.5, 11)
            elif self._task_name in (
                "swingup_vs_adversary",
                "swingup_sparse_vs_adversary",
            ):
                first_metric_range = np.linspace(0.05, 0.15, 11)
                second_metric_range = np.linspace(0.5, 1.5, 11)
            else:
                first_metric_range = np.array([])
                second_metric_range = np.array([])
        elif self._domain_name == "cheetah_two_players":
            first_metric_range = np.linspace(3, 9, 11)
            second_metric_range = np.linspace(0.1, 1.9, 11)
        elif self._domain_name == "hopper_two_players":
            first_metric_range = np.linspace(1, 9, 11)
            second_metric_range = np.linspace(0.1, 1.9, 11)
        elif self._domain_name == "pendulum_two_players":
            first_metric_range = np.linspace(0.05, 0.15, 11)
            second_metric_range = np.linspace(0.5, 1.5, 11)
        elif self._domain_name == "quadruped_two_players":
            first_metric_range = np.linspace(65, 75, 11)
            second_metric_range = np.linspace(0.95, 1.05, 11)
        elif self._domain_name == "reacher_two_players":
            first_metric_range = np.linspace(0.02, 0.06, 11)
            second_metric_range = np.linspace(0.02, 0.05, 11)
        elif self._domain_name == "walker_two_players":
            first_metric_range = np.linspace(5, 15, 11)
            second_metric_range = np.linspace(0.1, 1.3, 11)
        else:
            first_metric_range = np.array([])
            second_metric_range = np.array([])
        self.metric_ranges = [first_metric_range, second_metric_range]

    def adjust_steps_per_fit(self, agent):
        # because SAC is intensive, fit sac agent every 3 steps
        if agent == "trpo":
            self.n_steps_per_fit = self.n_steps_per_iteration
        elif agent == "sac":
            self.n_steps_per_fit = 3
        else:
            raise ValueError("Invalid agent type! Only 'trpo' or 'sac' allowed.")

    def log_params(self):
        """
        Put all the experiment parameters into the Experiment_<seed> log file
        """
        log_params_dict = {}
        log_params_dict.update(self.__dict__)
        msg = "\n" + "".join(f"{k}: {v}\n" for (k, v) in log_params_dict.items())
        self.exp_logger.info(msg)

    @abstractmethod
    def train_protagonist(self):
        raise NotImplementedError

    def train_worst_adversary(self, protagonist, existing_adversary=None):
        """
        Train an adversary to maximally disrupt a trained protagonist
        """
        if existing_adversary and self.n_total_iterations_worst_adversary == 0:
            return existing_adversary

        def setup():
            mdp = self.provide_mdp()
            if type(self.new_adv_max_force) == float:
                self.update_adversary(mdp, self.new_adv_max_force)

            adversary = SetupAgent(
                self.agent, mdp.info, idx_agent=1, use_cuda=self.use_cuda
            )
            worst_adv_logger = Logger(
                log_name="Worst_Adversary",
                results_dir=Path(self.results_dir) / "Logging",
                log_console=True,
                seed=self.seed,
                console_log_level=30,
            )
            adversary.set_logger(worst_adv_logger)

            agents = [protagonist, adversary]

            core = self.provide_core("multi-agent", agents, mdp)

            return core, worst_adv_logger

        core, worst_adv_logger = setup()

        # Train worst adversary
        mean_reward_vs_worst_adversary_progress = []
        for i in trange(self.n_total_iterations_worst_adversary, leave=False):
            # Optimization of adversary
            for _ in range(self.n_iterations_per_agent):
                core.learn(
                    n_steps=self.n_steps_per_iteration,
                    n_episodes=self.n_episodes_per_iteration,
                    n_steps_per_fit_per_agent=self.get_n_steps_per_fit_per_agent(
                        len(core.agent), idx_agent=1
                    ),
                    quiet=False,
                    render=self.bool_render,
                )

            # Evaluation of iteration
            mean_reward_vs_worst_adversary = self.evaluate_vs_adversary(
                core.agent[0], core.agent[1], int(self.n_evaluation_episodes / 10)
            )
            mean_reward_vs_worst_adversary_progress.append(
                mean_reward_vs_worst_adversary
            )
            msg = "Experiment iteration {}:  \t Mean reward vs worst adversary: {}".format(
                i, mean_reward_vs_worst_adversary
            )
            worst_adv_logger.info(msg)

            # Save best agents
            worst_adv_logger.log_best_agent(
                core.agent[1], -mean_reward_vs_worst_adversary
            )

        self.save_worst_adversary(
            core.agent[1],
            Path(self.results_dir) / "Training_worst_adversary",
            self.seed,
            full_save=False,
        )

        # Extract data
        data = {}

        # Mean reward per iteration
        data[
            "exp_" + str(self.seed) + "_mean_reward_vs_worst_adversary_per_iteration"
        ] = mean_reward_vs_worst_adversary_progress

        # Adversary data
        data[
            "exp_" + str(self.seed) + "_worst_adv_temp_per_training_step"
        ] = core.agent[1].temperature_data
        data[
            "exp_" + str(self.seed) + "_worst_adv_batch_mean_entropy_per_training_step"
        ] = core.agent[1].entropy_data
        data[
            "exp_" + str(self.seed) + "_worst_adv_actor_loss_per_training_step"
        ] = core.agent[1].actor_loss_data
        data[
            "exp_" + str(self.seed) + "_worst_adv_critic_loss_per_training_step"
        ] = core.agent[1].critic_loss_data

        self.save_data(Path(self.results_dir) / "Training_worst_adversary", **data)

        return core.agent[1]

    def evaluate(self, protagonist, worst_adversary):
        """
        Evaluate the performance of the protagonist across a number of measures (against no adversary, against a specific adversary,
        and across the robustness metrics)
        """
        mean_reward_vs_worst_adversary = self.evaluate_vs_adversary(
            protagonist, worst_adversary, n_episodes=self.n_evaluation_episodes
        )
        mean_reward_without_adversary = self.evaluate_without_adversary(
            protagonist, n_episodes=self.n_evaluation_episodes
        )
        mean_reward_first_metric = self.evaluate_robustness(
            protagonist,
            idx_metric=0,
            n_episodes_per_metric_value=self.n_evaluation_episodes_per_metric_value,
        )
        mean_reward_second_metric = self.evaluate_robustness(
            protagonist,
            idx_metric=1,
            n_episodes_per_metric_value=self.n_evaluation_episodes_per_metric_value,
        )
        mean_reward_both_metrics = self.evaluate_robustness(
            protagonist,
            idx_metric=-1,
            n_episodes_per_metric_value=self.n_evaluation_episodes_per_metric_value,
        )

        # Save data
        data = {}
        data[
            "exp_" + str(self.seed) + "_mean_reward_vs_worst_adversary"
        ] = mean_reward_vs_worst_adversary
        data[
            "exp_" + str(self.seed) + "_mean_reward_without_adversary"
        ] = mean_reward_without_adversary
        data[
            "exp_" + str(self.seed) + "_mean_reward_first_metric"
        ] = mean_reward_first_metric
        data[
            "exp_" + str(self.seed) + "_mean_reward_second_metric"
        ] = mean_reward_second_metric
        data[
            "exp_" + str(self.seed) + "_mean_reward_robustness"
        ] = mean_reward_both_metrics

        self.save_data(Path(self.results_dir) / "Evaluation", **data)

    def evaluate_vs_adversary(self, protagonist, adversary, n_episodes):
        def setup():
            mdp = self.provide_mdp()
            if type(self.new_adv_max_force) == float:
                self.update_adversary(mdp, self.new_adv_max_force)

            agents = [protagonist, adversary]

            collect_dataset = CollectDataset()
            callbacks = collect_dataset
            core = self.provide_core(
                "multi-agent", agents, mdp, callback_step=callbacks
            )

            return core

        core = setup()

        core.evaluate(n_episodes=n_episodes, render=self.bool_render)

        # Extract data
        cumulative_reward_per_episode = compute_J(core.callback_step.get(), idx_agent=0)
        mean_reward = np.mean(cumulative_reward_per_episode)

        # Quadruped data

        return mean_reward

    def evaluate_without_adversary(self, protagonist, n_episodes):
        def setup():
            mdp = self.provide_mdp()
            if type(self.new_adv_max_force) == float:
                self.update_adversary(mdp, self.new_adv_max_force)

            constant_adversary = SetupAgent("constant", mdp.info, idx_agent=1)
            agents = [protagonist, constant_adversary]

            collect_dataset = CollectDataset()
            callbacks = collect_dataset
            core = self.provide_core(
                "multi-agent", agents, mdp, callback_step=callbacks
            )

            return core

        core = setup()

        core.evaluate(n_episodes=n_episodes, render=self.bool_render)

        # Extract data
        cumulative_reward_per_episode = compute_J(core.callback_step.get(), idx_agent=0)
        mean_reward = np.mean(cumulative_reward_per_episode)

        return mean_reward

    def evaluate_vs_worst_adversary(self, protagonist, worst_adversary, n_episodes):
        mean_reward_vs_worst_adversary = self.evaluate_vs_adversary(
            protagonist, worst_adversary, n_episodes
        )
        # Save data
        data = {}
        data[
            "exp_" + str(self.seed) + "_mean_reward_vs_true_worst_adversary"
        ] = mean_reward_vs_worst_adversary

        self.save_data(Path(self.results_dir) / "Evaluation", **data)

    def evaluate_vs_perfect_adversary(self, protagonist, n_episodes):
        def setup():
            mdp = self.provide_mdp()
            if type(self.new_adv_max_force) == float:
                self.update_adversary(mdp, self.new_adv_max_force)

            constant_policy = ConstantPolicy(
                mdp.info.action_space[1].shape, constant_value=self.new_adv_max_force
            )
            adversary = MultiPlayerAgent(mdp.info, constant_policy, 1)

            agents = [protagonist, adversary]

            collect_dataset = CollectDataset()
            callbacks = collect_dataset
            core = self.provide_core(
                "multi-agent", agents, mdp, callback_step=callbacks
            )

            return core

        core = setup()
        core.evaluate(n_episodes=n_episodes, render=self.bool_render)

        # Extract data
        cumulative_reward_per_episode = compute_J(core.callback_step.get(), idx_agent=0)
        mean_reward = np.mean(cumulative_reward_per_episode)

        return mean_reward

    def evaluate_robustness(self, protagonist, idx_metric, n_episodes_per_metric_value):
        """
        Evaluate return across robustness metrics.
        idx_metric: 1 for first metric, 2 for second metric, -1 for both
        """

        def setup():
            mdp = self.provide_mdp()
            if type(self.new_adv_max_force) == float:
                self.update_adversary(mdp, self.new_adv_max_force)

            constant_adversary = SetupAgent("constant", mdp.info, idx_agent=1)
            agents = [protagonist, constant_adversary]

            collect_dataset = CollectDataset()
            callbacks = collect_dataset
            core = self.provide_core(
                "multi-agent", agents, mdp, callback_step=callbacks
            )

            return core

        core = setup()

        # Evaluate changing metrics
        if idx_metric == -1:
            mean_reward_per_metric_value = np.zeros(
                (self.metric_ranges[0].shape[0], self.metric_ranges[1].shape[0], 3)
            )

            for first_metric_idx, first_metric_value in enumerate(
                self.metric_ranges[0]
            ):
                for second_metric_idx, second_metric_value in enumerate(
                    self.metric_ranges[1]
                ):
                    core.mdp.env.physics.change_first_metric(first_metric_value)
                    core.mdp.env.physics.change_second_metric(second_metric_value)
                    core.evaluate(
                        n_episodes=n_episodes_per_metric_value, render=self.bool_render
                    )

                    # Extract data
                    cumulative_reward_per_episode = compute_J(
                        core.callback_step.get(), idx_agent=0
                    )
                    mean_reward = np.mean(cumulative_reward_per_episode)
                    mean_reward_per_metric_value[first_metric_idx][
                        second_metric_idx
                    ] = [
                        first_metric_value,
                        second_metric_value,
                        mean_reward,
                    ]
                    core.callback_step.clean()

        elif idx_metric == 0 or idx_metric == 1:
            mean_reward_per_metric_value = []
            for metric_value in self.metric_ranges[idx_metric]:
                if idx_metric == 0:
                    core.mdp.env.physics.change_first_metric(metric_value)
                elif idx_metric == 1:
                    core.mdp.env.physics.change_second_metric(metric_value)
                core.evaluate(
                    n_episodes=n_episodes_per_metric_value, render=self.bool_render
                )

                # Extract data
                cumulative_reward_per_episode = compute_J(
                    core.callback_step.get(), idx_agent=0
                )
                mean_reward = np.mean(cumulative_reward_per_episode)
                mean_reward_per_metric_value.append([metric_value, mean_reward])
                core.callback_step.clean()

        else:
            raise ValueError("Invalid robustness metric! Only 0, 1, -1 allowed.")

        return mean_reward_per_metric_value

    def evaluate_quadruped_success_rate(self, protagonist, n_episodes):
        def setup():
            mdp = self.provide_mdp()
            if type(self.new_adv_max_force) == float:
                self.update_adversary(mdp, self.new_adv_max_force)

            constant_adversary = SetupAgent("constant", mdp.info, idx_agent=1)
            agents = [protagonist, constant_adversary]

            collect_dataset = CollectDataset()
            callbacks = collect_dataset
            core = self.provide_core(
                "multi-agent", agents, mdp, callback_step=callbacks
            )

            return core

        core = setup()

        core.evaluate(n_episodes=n_episodes, render=self.bool_render)

        # Extract data
        distances, successes = compute_quadruped_success_rate(
            core.callback_step.get(), idx_agent=0
        )
        success_rate = np.mean(successes)

        return success_rate

    def get_n_steps_per_fit_per_agent(self, num_agents, idx_agent):
        """
        method to get n_steps_per_fit_per_agent for core.learn()
        idx_agent = agent which is currently learning
        """
        if num_agents == 2:
            n_steps_per_fit_per_agent = [None] * 2
            if self.n_steps_per_iteration:
                n_steps_per_fit_per_agent[idx_agent] = self.n_steps_per_fit
                n_steps_per_fit_per_agent[1 - idx_agent] = (
                    self.n_steps_per_iteration + 1
                )
            elif self.n_episodes_per_iteration:
                n_steps_per_fit_per_agent[idx_agent] = self.n_steps_per_fit
                n_steps_per_fit_per_agent[1 - idx_agent] = (
                    self.n_episodes_per_iteration * self._horizon + 1
                )
            else:
                raise ValueError(
                    "n_steps_per_iteration or n_episodes_per_iteration must not be None!"
                )
        elif num_agents == 1:
            if self.n_steps_per_iteration or self.n_episodes_per_iteration:
                n_steps_per_fit_per_agent = [self.n_steps_per_fit]
            else:
                raise ValueError(
                    "n_steps_per_iteration or n_episodes_per_iteration must not be None!"
                )
        return n_steps_per_fit_per_agent

    @staticmethod
    def update_adversary(mdp, new_adv_max_force):
        # update of dm_control physics
        mdp.env.task.update_adversary(mdp.env.physics, new_adv_max_force)
        # update of mdp_info
        adv_action_spec_shape = mdp.env.physics.adv_action_spec.shape
        adv_max_forces = np.ones(adv_action_spec_shape) * new_adv_max_force
        adv_min_forces = -adv_max_forces
        if "Wind" in type(mdp.env.task).__name__:
            adv_min_forces = 0 * adv_max_forces

        mdp.info.action_space[1] = Box(low=adv_min_forces, high=adv_max_forces)

    def provide_mdp(self):
        mdp = dm_control_env.DMControl(**self.environment_params)
        return mdp

    @staticmethod
    def provide_core(
        type,
        agents,
        mdp,
        callbacks_fit=None,
        callback_step=None,
        state_preprocessors=None,
    ):
        if type == "multi-agent":
            core = MultiAgentCore(
                agents, mdp, callbacks_fit, callback_step, state_preprocessors
            )
        elif type == "self-paced":
            core = SelfPacedRLCore(
                agents, mdp, callbacks_fit, callback_step, state_preprocessors
            )
        elif type == "mas":
            core = MASCore(
                agents, mdp, callbacks_fit, callback_step, state_preprocessors
            )
        elif type == "force":
            core = SelfPacedForceCore(
                agents, mdp, callbacks_fit, callback_step, state_preprocessors
            )
        else:
            assert False, "Unknown type of mushroomRL core provided"
        return core

    @staticmethod
    def load_agent(agent_path):
        agent = Serializable.load(agent_path)
        return agent

    @staticmethod
    def save_baseline_agent(agent, results_dir, seed, full_save=False):
        filename = "exp_" + str(seed) + "_baseline.zip"
        agent.save(Path(results_dir) / filename, full_save=full_save)

    @staticmethod
    def save_protagonist(agent, results_dir, seed, full_save=False):
        filename = "exp_" + str(seed) + "_protagonist.zip"

        agent.save(Path(results_dir) / filename, full_save=full_save)

    @staticmethod
    def save_worst_adversary(agent, results_dir, seed, full_save=False):
        filename = "exp_" + str(seed) + "_worst_adversary.zip"
        agent.save(Path(results_dir) / filename, full_save=full_save)

    @staticmethod
    def save_agents(agents, results_dir, seed, full_save=False):
        protagonist_filename = "exp_" + str(seed) + "_protagonist.zip"
        adversary_filename = "exp_" + str(seed) + "_adversary.zip"

        agents[0].save(Path(results_dir) / protagonist_filename, full_save=full_save)
        agents[1].save(Path(results_dir) / adversary_filename, full_save=full_save)

    @staticmethod
    def save_data(results_dir, **kwargs):
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        for name, data in kwargs.items():
            filename = name + ".npy"
            path = Path(results_dir) / filename

            np.save(str(path), data)
