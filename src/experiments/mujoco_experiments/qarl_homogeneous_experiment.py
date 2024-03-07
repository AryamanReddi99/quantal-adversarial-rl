from pathlib import Path

import numpy as np
from mushroom_rl.core.logger.logger import Logger
from mushroom_rl.utils.callbacks import CollectDataset
from mushroom_rl.utils.spaces import Box
from mushroom_rl_extensions.agents.create_agent import SetupAgent
from mushroom_rl_extensions.utils import dataset
from mushroom_rl_extensions.utils.dataset import compute_J
from teachers.self_paced_gamma_teacher_fixed_beta import (
    SelfPacedGammaTemperatureTeacherFixedBeta,
)
from tqdm import trange
from utils.buffer import Buffer
from utils.state_preprocessor import StateVariableConcatenation

from .qarl_experiment import QARLExperiment


class QARLHomogeneousExperiment(QARLExperiment):
    """
    Runs an adversarial experiment that trains an agent against an adversary with a self-paced temperature curriculum (QARL).
    Both agents' temperatures are determined by the temperature curriculum.

    initial_conc: concentration of gamma distribution at beginning of curriculum
    target_conc: target concentration of gamma distribution
    fixed_rate: rate of gamma distribution throughout training
    conc_lower_bound: lower bound of gamma concentration
    max_kl: maximum kl divergence between subsequent gamma distributions
    performance_lower_bound: minimum score the agent must achieve for curriculum to move towards target
    target_kl_threshold: threshold of current temperature distribution from target before curriculum stops
    temperature_bounds: scalar temperature bounds
    n_samples_for_dist_update: number of rollouts needed for update of temperature distribution
    """

    def train_protagonist(self):
        def setup():
            mdp = self.provide_mdp()
            mdp.info.observation_space = self.extend_observation_space(mdp)
            if type(self.new_adv_max_force) == float:
                self.update_adversary(mdp, self.new_adv_max_force)
            protagonist = SetupAgent(
                "adaptive_temp_" + self.agent,
                mdp.info,
                idx_agent=0,
                use_cuda=self.use_cuda,
            )
            prot_logger = Logger(
                log_name="Protagonist",
                results_dir=Path(self.results_dir) / "Logging",
                log_console=True,
                seed=self.seed,
                console_log_level=30,
            )
            protagonist.set_logger(prot_logger)

            adversary = SetupAgent(
                "adaptive_temp_" + self.agent,
                mdp.info,
                idx_agent=1,
                use_cuda=self.use_cuda,
            )
            adv_logger = Logger(
                log_name="Adversary",
                results_dir=Path(self.results_dir) / "Logging",
                log_console=True,
                seed=self.seed,
                console_log_level=30,
            )
            adversary.set_logger(adv_logger)

            teacher = self.provide_teacher(type="gamma_fixed_beta")
            teacher_logger = Logger(
                log_name="Teacher",
                results_dir=Path(self.results_dir) / "Logging",
                log_console=True,
                seed=self.seed,
                console_log_level=30,
            )
            teacher.set_logger(teacher_logger)

            agents = [protagonist, adversary]

            collect_dataset = CollectDataset()
            callbacks = collect_dataset
            preprocessor = StateVariableConcatenation()
            core = self.provide_core(
                "self-paced",
                agents,
                mdp,
                callback_step=callbacks,
                state_preprocessors=[preprocessor],
            )
            temp_perf_buffer = Buffer(2, 100, reset_on_query=True)

            return (
                core,
                teacher,
                temp_perf_buffer,
                prot_logger,
                adv_logger,
                teacher_logger,
            )

        (
            core,
            teacher,
            temp_perf_buffer,
            prot_logger,
            adv_logger,
            teacher_logger,
        ) = setup()

        # Train agents
        mean_reward_vs_adversary_progress = []
        mean_reward_without_adversary_progress = []
        sampled_temperatures_list = []
        for i in trange(self.n_total_iterations, leave=False):
            # Temperature sampling for iteration
            if self.n_steps_per_iteration:
                raise NotImplementedError
            elif self.n_episodes_per_iteration:
                sampled_temperatures = self.sample_n_temperatures(
                    teacher, self.n_episodes_per_iteration
                )
            else:
                raise ValueError(
                    "n_steps_per_iteration or n_episodes_per_iteration must not be None!"
                )

            # Temperature Stats
            sampled_temperatures_list.append(np.mean(sampled_temperatures))
            core.set_variables_for_training(sampled_temperatures)
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
            # Optimization of protagonist
            for _ in range(self.n_iterations_per_agent):
                core.learn(
                    n_steps=self.n_steps_per_iteration,
                    n_episodes=self.n_episodes_per_iteration,
                    n_steps_per_fit_per_agent=self.get_n_steps_per_fit_per_agent(
                        len(core.agent), idx_agent=0
                    ),
                    quiet=False,
                    render=self.bool_render,
                )
            # Evaluation of iteration
            evaluation_temp = teacher.temp_dist.mean()

            # Vs adversary
            mean_reward_vs_adversary = self.evaluate_vs_adversary(
                core.agent[0],
                core.agent[1],
                n_episodes=int(self.n_evaluation_episodes / 10),
                evaluation_temp=evaluation_temp,
            )
            mean_reward_vs_adversary_progress.append(mean_reward_vs_adversary)
            msg_vs_adv = (
                "Experiment iteration {}:  \t Mean reward vs adversary: {}".format(
                    i, mean_reward_vs_adversary
                )
            )
            prot_logger.info(msg_vs_adv)

            # Without adversary
            mean_reward_without_adversary = self.evaluate_without_adversary(
                core.agent[0],
                n_episodes=int(self.n_evaluation_episodes / 10),
                evaluation_temp=evaluation_temp,
            )
            mean_reward_without_adversary_progress.append(mean_reward_without_adversary)
            msg_without_adv = (
                "Experiment iteration {}:  \t Mean reward without adversary: {}".format(
                    i, mean_reward_without_adversary
                )
            )
            prot_logger.info(msg_without_adv)

            # Update of temperature distribution
            temperatures, sample_returns = self.extract_temp_and_sample_return(
                core.callback_step.get(), 0, core.mdp.info.gamma
            )
            temp_perf_buffer.update_buffer([temperatures, sample_returns])
            core.callback_step.clean()
            if temp_perf_buffer.__len__() >= self.n_samples_for_dist_update:
                temps_perfs = temp_perf_buffer.read_buffer(reset=True)
                teacher.update_temperature_distribution(
                    temps_perfs[0], temps_perfs[1], i
                )

            # Save best agents
            prot_logger.log_best_agent(core.agent[0], mean_reward_vs_adversary)
            adv_logger.log_best_agent(core.agent[1], -mean_reward_vs_adversary)

        self.save_agents(
            core.agent,
            Path(self.results_dir) / "Training",
            self.seed,
            full_save=False,
        )

        ## Data recording
        data = {}
        # Mean reward per iteration
        data[
            "exp_" + str(self.seed) + "_mean_reward_vs_adversary_per_iteration"
        ] = mean_reward_vs_adversary_progress
        data[
            "exp_" + str(self.seed) + "_mean_reward_without_adversary_per_iteration"
        ] = mean_reward_without_adversary_progress

        # Teacher data per iteration
        data[
            "exp_" + str(self.seed) + "_teacher_temp_conc_per_iteration"
        ] = teacher.conc_temperature_data
        data[
            "exp_" + str(self.seed) + "_teacher_mean_temp_per_iteration"
        ] = sampled_temperatures_list

        # Protagonist data
        data["exp_" + str(self.seed) + "_prot_temp_per_training_step"] = core.agent[
            0
        ].temperature_data
        data[
            "exp_" + str(self.seed) + "_prot_batch_mean_entropy_per_training_step"
        ] = core.agent[0].entropy_data
        data[
            "exp_" + str(self.seed) + "_prot_actor_loss_per_training_step"
        ] = core.agent[0].actor_loss_data
        data[
            "exp_" + str(self.seed) + "_prot_critic_loss_per_training_step"
        ] = core.agent[0].critic_loss_data

        # Adversary data
        data["exp_" + str(self.seed) + "_adv_temp_per_training_step"] = core.agent[
            1
        ].temperature_data
        data[
            "exp_" + str(self.seed) + "_adv_batch_mean_entropy_per_training_step"
        ] = core.agent[1].entropy_data
        data[
            "exp_" + str(self.seed) + "_adv_actor_loss_per_training_step"
        ] = core.agent[1].actor_loss_data
        data[
            "exp_" + str(self.seed) + "_adv_critic_loss_per_training_step"
        ] = core.agent[1].critic_loss_data

        self.save_data(Path(self.results_dir) / "Training", **data)

        return core.agent[0], core.agent[1]

    def train_worst_adversary(self, protagonist, existing_adversary=None):
        if existing_adversary and self.n_total_iterations_worst_adversary == 0:
            return existing_adversary
        # Protagonist behaves as it would expect the worst adversary
        adv_temperature = 0.0

        def setup():
            mdp = self.provide_mdp()
            mdp.info.observation_space = self.extend_observation_space(mdp)
            if type(self.new_adv_max_force) == float:
                self.update_adversary(mdp, self.new_adv_max_force)

            adversary = SetupAgent(
                self.agent,
                mdp.info,
                idx_agent=1,
                use_cuda=self.use_cuda,
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

            preprocessor = StateVariableConcatenation()
            core = self.provide_core(
                "self-paced", agents, mdp, state_preprocessors=[preprocessor]
            )

            return core, worst_adv_logger

        core, worst_adv_logger = setup()

        # Train worst adversary
        mean_reward_vs_worst_adversary_progress = []

        # Set adversary temperatures for training
        if self.n_steps_per_iteration:
            raise NotImplementedError
        elif self.n_episodes_per_iteration:
            temperatures = np.ones(self.n_episodes_per_iteration) * adv_temperature
        else:
            raise ValueError(
                "n_steps_per_iteration or n_episodes_per_iteration must not be None!"
            )
        core.set_variables_for_training(temperatures)
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
                core.agent[0],
                core.agent[1],
                n_episodes=int(self.n_evaluation_episodes / 10),
                evaluation_temp=adv_temperature,
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
        data[
            "exp_" + str(self.seed) + "_mean_reward_vs_worst_adversary_per_iteration"
        ] = mean_reward_vs_worst_adversary_progress
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

    def evaluate_vs_adversary(
        self, protagonist, adversary, n_episodes, evaluation_temp=0.0
    ):
        def setup():
            mdp = self.provide_mdp()
            mdp.info.observation_space = self.extend_observation_space(mdp)
            if type(self.new_adv_max_force) == float:
                self.update_adversary(mdp, self.new_adv_max_force)
            agents = [protagonist, adversary]

            collect_dataset = CollectDataset()
            callbacks = collect_dataset
            preprocessor = StateVariableConcatenation()
            core = self.provide_core(
                "self-paced",
                agents,
                mdp,
                callback_step=callbacks,
                state_preprocessors=[preprocessor],
            )

            return core

        core = setup()

        temperatures = np.ones(n_episodes) * evaluation_temp
        core.set_variables_for_training(temperatures)
        core.evaluate(n_episodes=n_episodes, render=self.bool_render)

        # Extract data
        reward_per_episode = dataset.compute_J(core.callback_step.get(), idx_agent=0)
        mean_reward = np.mean(reward_per_episode)

        return mean_reward

    def evaluate_without_adversary(self, protagonist, n_episodes, evaluation_temp=0.0):
        def setup():
            mdp = self.provide_mdp()
            mdp.info.observation_space = self.extend_observation_space(mdp)
            if type(self.new_adv_max_force) == float:
                self.update_adversary(mdp, self.new_adv_max_force)

            constant_adversary = SetupAgent("constant", mdp.info, idx_agent=1)
            agents = [protagonist, constant_adversary]

            collect_dataset = CollectDataset()
            callbacks = collect_dataset
            preprocessor = StateVariableConcatenation()
            core = self.provide_core(
                "self-paced",
                agents,
                mdp,
                callback_step=callbacks,
                state_preprocessors=[preprocessor],
            )

            return core

        core = setup()

        temperatures = np.ones(n_episodes) * evaluation_temp
        core.set_variables_for_training(temperatures)
        core.evaluate(n_episodes=n_episodes, render=self.bool_render)

        # Extract data
        reward_per_episode = dataset.compute_J(core.callback_step.get(), idx_agent=0)
        mean_reward = np.mean(reward_per_episode)

        return mean_reward

    def evaluate_robustness(
        self, protagonist, idx_metric, n_episodes_per_metric_value, evaluation_temp=0.0
    ):
        """
        Evaluate return across robustness metrics.
        idx_metric: 1 for first metric, 2 for second metric, -1 for both
        """

        def setup():
            mdp = self.provide_mdp()
            mdp.info.observation_space = self.extend_observation_space(mdp)
            if type(self.new_adv_max_force) == float:
                self.update_adversary(mdp, self.new_adv_max_force)
            constant_adversary = SetupAgent("constant", mdp.info, idx_agent=1)
            agents = [protagonist, constant_adversary]

            collect_dataset = CollectDataset()
            callbacks = collect_dataset
            preprocessor = StateVariableConcatenation()
            core = self.provide_core(
                "self-paced",
                agents,
                mdp,
                callback_step=callbacks,
                state_preprocessors=[preprocessor],
            )

            return core

        core = setup()
        temperatures = np.ones(n_episodes_per_metric_value) * evaluation_temp
        core.set_variables_for_training(temperatures)

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

    def provide_teacher(self, type):
        if type == "gamma_fixed_beta":
            teacher = SelfPacedGammaTemperatureTeacherFixedBeta(
                self.initial_conc,
                self.target_conc,
                self.fixed_rate,
                self.max_kl,
                self.performance_lower_bound,
                self.conc_lower_bound,
                self.target_kl_threshold,
                self.temperature_bounds,
            )
        else:
            raise ValueError("Unknown teacher provided!")

        return teacher

    @staticmethod
    def extend_observation_space(mdp):
        """
        Extend state space by one dimension to include temperature scalar
        """
        low = mdp.info.observation_space.low
        high = mdp.info.observation_space.high
        old_shape = mdp.info.observation_space.shape
        # adding temperature to observation space shape
        new_shape = tuple((list(old_shape)[0] + 1,))

        return Box(low=low, high=high, shape=new_shape)

    @staticmethod
    def sample_n_temperatures(teacher, n):
        sampled_temperatures = np.zeros(n)
        for i in range(n):
            sampled_temperatures[i] = teacher.sample()

        return sampled_temperatures

    @staticmethod
    def extract_temp_and_sample_return(data, idx_agent, gamma):
        temperatures = list()
        for i in range(len(data)):
            if data[i][-1] or i == len(data) - 1:
                temperatures.append(data[i][0][-1])

        sample_returns = dataset.compute_J(data, idx_agent, gamma)

        return temperatures, sample_returns
