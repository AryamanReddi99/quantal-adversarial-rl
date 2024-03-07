from pathlib import Path

import numpy as np
from mushroom_rl.core.logger.logger import Logger
from mushroom_rl.utils.callbacks import CollectDataset
from mushroom_rl_extensions.agents.create_agent import SetupAgent
from mushroom_rl_extensions.utils.dataset import compute_J
from tqdm import trange

from .abstract_experiment import AbstractExperiment


class MASExperiment(AbstractExperiment):
    """
    MAS budget curriculum algorithm.
    final_update_iteration: the iteration fraction at which we want the curriculum to stop updating (max budget reached). Budget will be updated evenly between these iterations.
    initial_update_iteration: the iteration fraction at which the budget starts
    """

    def __init__(
        self,
        initial_update_iteration: int = 0.2,
        final_update_iteration: int = 0.8,
        **kwargs,
    ):
        self.initial_update_iteration = initial_update_iteration
        self.final_update_iteration = final_update_iteration
        if "new_adv_max_force" in kwargs:
            self.target_budget = kwargs["new_adv_max_force"]
        else:
            self.target_budget = 1.0
        super().__init__(**kwargs)

    def train_protagonist(self):
        def setup():
            mdp = self.provide_mdp()
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

            adversary = SetupAgent("mas", mdp.info, idx_agent=1, use_cuda=self.use_cuda)
            adv_logger = Logger(
                log_name="Adversary",
                results_dir=Path(self.results_dir) / "Logging",
                log_console=True,
                seed=self.seed,
                console_log_level=30,
            )
            adversary.set_logger(adv_logger)

            agents = [protagonist, adversary]

            core = self.provide_core("mas", agents, mdp)

            return core, prot_logger, adv_logger

        core, prot_logger, adv_logger = setup()

        # Budget curriculum
        warmup_phase = np.zeros(
            int(self.initial_update_iteration * self.n_total_iterations)
        )
        print(
            int(
                (round(self.final_update_iteration - self.initial_update_iteration, 2))
                * self.n_total_iterations
            ),
        )
        training_phase = np.linspace(
            0,
            self.target_budget,
            int(
                (round(self.final_update_iteration - self.initial_update_iteration, 2))
                * self.n_total_iterations
            ),
        )
        final_phase = (
            np.ones(
                int(round(1 - self.final_update_iteration, 2) * self.n_total_iterations)
            )
            * self.target_budget
        )
        budget_curriculum = np.concatenate(
            (warmup_phase, training_phase, final_phase), axis=0
        )
        assert len(budget_curriculum) == self.n_total_iterations

        # Train agents
        mean_reward_vs_adversary_progress = []
        mean_reward_without_adversary_progress = []
        for i in trange(self.n_total_iterations, leave=False):
            # Set budget
            core.agent[1].set_budget(budget_curriculum[i])

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
            # Vs adversary
            mean_reward_vs_adversary = self.evaluate_vs_adversary(
                core.agent[0],
                core.agent[1],
                n_episodes=int(self.n_evaluation_episodes / 10),
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
                core.agent[0], n_episodes=int(self.n_evaluation_episodes / 10)
            )
            mean_reward_without_adversary_progress.append(mean_reward_without_adversary)
            msg_without_adv = (
                "Experiment iteration {}:  \t Mean reward without adversary: {}".format(
                    i, mean_reward_without_adversary
                )
            )
            prot_logger.info(msg_without_adv)

            # Save best agents
            prot_logger.log_best_agent(core.agent[0], mean_reward_vs_adversary)

        self.save_protagonist(
            core.agent[0],
            Path(self.results_dir) / "Training",
            self.seed,
            full_save=False,
        )

        ## Extract data
        data = {}
        # Mean reward per iteration
        data[
            "exp_" + str(self.seed) + "_mean_reward_vs_adversary_per_iteration"
        ] = mean_reward_vs_adversary_progress
        data[
            "exp_" + str(self.seed) + "_mean_reward_without_adversary_per_iteration"
        ] = mean_reward_without_adversary_progress

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
        data[
            "exp_" + str(self.seed) + "_mas_action_norm_per_training_step"
        ] = core.agent[1].policy.mas_action_norm_data

        # Curriculum data
        data["exp_" + str(self.seed) + "_budget_per_iteration"] = budget_curriculum

        self.save_data(Path(self.results_dir) / "Training", **data)

        return core.agent[0], core.agent[1]

    def train_worst_adversary(self, protagonist, existing_adversary=None):
        mdp = self.provide_mdp()
        if type(self.new_adv_max_force) == float:
            self.update_adversary(mdp, self.new_adv_max_force)
        adversary = SetupAgent(
            "mas",
            mdp.info,
            idx_agent=1,
            use_cuda=self.use_cuda,
            budget=self.target_budget,
        )
        return adversary

    def evaluate_vs_adversary(self, protagonist, adversary, n_episodes):
        def setup():
            mdp = self.provide_mdp()
            if type(self.new_adv_max_force) == float:
                self.update_adversary(mdp, self.new_adv_max_force)

            agents = [protagonist, adversary]

            collect_dataset = CollectDataset()
            callbacks = collect_dataset
            core = self.provide_core("mas", agents, mdp, callback_step=callbacks)

            return core

        core = setup()

        core.evaluate(n_episodes=n_episodes, render=self.bool_render)

        # Extract data
        cumulative_reward_per_episode = compute_J(core.callback_step.get(), idx_agent=0)
        mean_reward = np.mean(cumulative_reward_per_episode)

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
            core = self.provide_core("mas", agents, mdp, callback_step=callbacks)

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
            core = self.provide_core("mas", agents, mdp, callback_step=callbacks)

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
