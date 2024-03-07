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
from teachers.self_paced_point_teacher import SelfPacedPointTemperatureTeacher
from tqdm import trange
from utils.buffer import Buffer
from utils.state_preprocessor import StateVariableConcatenation

from .qarl_experiment import QARLExperiment


class QARLPointExperiment(QARLExperiment):
    """
    Runs an adversarial experiment that trains an agent against an adversary with a self-paced temperature curriculum (QARL).
    The temperature is a single value that is optimised at each step of the curriculum (not sampled from a distribution)
    and is optimised using MSE minimisation subject to the usual QARL constraints on D_KL and performance threshold.

    initial_temp: temperature at beginning of curriculum
    target_temp: target temperature of curriculum
    temp_lower_bound: lower bound of temperature
    max_mse: maximum mse divergence between subsequent temperatures
    performance_lower_bound: minimum score the agent must achieve for curriculum to move towards target
    target_mse_threshold: mse threshold of current temperature from target before curriculum stops
    n_samples_for_dist_update: number of rollouts needed for update of temperature distribution
    """

    def __init__(
        self,
        initial_temp: float = 0.05,
        target_temp: float = 1e-8,
        temp_lower_bound: float = 1e-8,
        max_mse: float = 0.000001,
        performance_lower_bound: int = 10,  # cheetah: 10, walker: 10, hopper: 1, cartpole: 10, quad:100
        target_mse_threshold: float = 1e-15,
        n_samples_for_dist_update: int = 30,
        **kwargs,
    ):
        self.initial_temp = initial_temp
        self.target_temp = target_temp
        self.temp_lower_bound = temp_lower_bound
        self.max_mse = max_mse
        self.performance_lower_bound = performance_lower_bound
        self.target_mse_threshold = target_mse_threshold
        self.n_samples_for_dist_update = n_samples_for_dist_update

        super().__init__(**kwargs)

    def train_protagonist(self):
        def setup():
            mdp = self.provide_mdp()
            mdp.info.observation_space = self.extend_observation_space(mdp)
            if type(self.new_adv_max_force) == float:
                self.update_adversary(mdp, self.new_adv_max_force)
            protagonist = SetupAgent(
                self.agent, mdp.info, idx_agent=0, use_cuda=self.use_cuda
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

            teacher = self.provide_teacher(type="point")
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
            evaluation_temp = teacher.temp

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
        elif type == "point":
            teacher = SelfPacedPointTemperatureTeacher(
                self.initial_temp,
                self.target_temp,
                self.max_mse,
                self.performance_lower_bound,
                self.temp_lower_bound,
                self.target_mse_threshold,
            )
        else:
            raise ValueError("Unknown teacher provided!")

        return teacher
