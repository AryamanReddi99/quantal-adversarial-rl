from pathlib import Path

import numpy as np
from mushroom_rl.core.logger.logger import Logger
from mushroom_rl.utils.callbacks import CollectDataset
from mushroom_rl.utils.spaces import Box
from mushroom_rl_extensions.agents.create_agent import SetupAgent
from mushroom_rl_extensions.utils.dataset import compute_J
from mushroom_rl_extensions.utils.functions import linear_bounded_function
from tqdm import trange
from utils.state_preprocessor import StateVariableConcatenation

from .qarl_experiment import QARLExperiment


class QARLLinearExperiment(QARLExperiment):
    """
    Runs an adversarial experiment that trains an agent against an adversary with a linear temperature curriculum.
    Temperature stays at initial_temp until descent_start_fraction*n_total_iterations iterations have passed
    Temperature then falls linearly until it reaches target_temp value after descent_end_fraction*n_total_iterations
    iterations have passed

    initial_temp: adversary temperature at beginning of curriculum
    target_temp: adversary temperature at end of curriculum
    descent_start_fraction: fraction of total iterations at which temperature starts falling
    descent_end_fraction: fraction of total iterations at which temperature stops falling
    """

    def __init__(
        self,
        initial_temp: float = 0.05,
        target_temp: float = 0.001,
        descent_start_fraction: float = 0,
        descent_end_fraction: float = 2 / 3,
        **kwargs,
    ):
        self.initial_temp = initial_temp
        self.target_temp = target_temp
        self.descent_start_fraction = descent_start_fraction
        self.descent_end_fraction = descent_end_fraction
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

            temp_logger = Logger(
                log_name="Temperature",
                results_dir=Path(self.results_dir) / "Logging",
                log_console=True,
                seed=self.seed,
                console_log_level=30,
            )

            adversary.set_logger(adv_logger)

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

            return (core, prot_logger, adv_logger, temp_logger)

        (core, prot_logger, adv_logger, temp_logger) = setup()

        # Train agents
        mean_reward_vs_adversary_progress = []
        mean_reward_without_adversary_progress = []
        sampled_temperatures_list = []
        for i in trange(self.n_total_iterations, leave=False):
            # Temperature sampling for iteration
            if self.n_steps_per_iteration:
                raise NotImplementedError
            elif self.n_episodes_per_iteration:
                sampled_temperatures = [
                    linear_bounded_function(
                        self.initial_temp,
                        self.target_temp,
                        self.descent_start_fraction * self.n_total_iterations,
                        self.descent_end_fraction * self.n_total_iterations,
                        i,
                    )
                    for _ in range(self.n_episodes_per_iteration)
                ]
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
            evaluation_temp = np.mean(sampled_temperatures)

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

            # Temperature logger
            msg_temp = "Experiment iteration {}:  \t Mean temp: {}".format(
                i, evaluation_temp
            )
            temp_logger.info(msg_temp)

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

        # Temperature data per iteration
        data[
            "exp_" + str(self.seed) + "_mean_temp_per_iteration"
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
