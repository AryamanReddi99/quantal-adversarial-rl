import numpy as np

from tqdm import trange
from pathlib import Path

from .abstract_experiment import AbstractExperiment
from mushroom_rl_extensions.agents.create_agent import SetupAgent

from mushroom_rl.core.logger.logger import Logger
from mushroom_rl.utils.callbacks import CollectDataset
from mushroom_rl_extensions.utils.dataset import compute_J


class SGLDExperiment(AbstractExperiment):
    """
    Runs a robust adversarial reinforcement learning training algorithm with a Langevin dynamics-based network optimizer
    Based on https://arxiv.org/abs/2002.06063
    """

    def train_protagonist(self):
        def setup():
            mdp = self.provide_mdp()
            if type(self.new_adv_max_force) == float:
                self.update_adversary(mdp, self.new_adv_max_force)

            protagonist = SetupAgent(
                self.agent + "_ld", mdp.info, idx_agent=0, use_cuda=self.use_cuda
            )
            prot_logger = Logger(
                log_name="Protagonist",
                results_dir=Path(self.results_dir) / "Logging",
                log_console=True,
                seed=self.seed,
                console_log_level=30,
            )
            protagonist.set_logger(prot_logger)

            constant_adversary = SetupAgent("constant", mdp.info, idx_agent=1)

            agents = [protagonist, constant_adversary]

            core = self.provide_core("multi-agent", agents, mdp)

            return core, prot_logger

        core, prot_logger = setup()

        # Train agents
        mean_reward_vs_adversary_progress = []
        mean_reward_without_adversary_progress = []
        for i in trange(self.n_total_iterations, leave=False):
            # Optimization of both agents
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
        data["exp_" + str(self.seed) + "_adv_temp_per_training_step"] = core.agent[
            0
        ].temperature_data_adversary
        data[
            "exp_" + str(self.seed) + "_adv_batch_mean_entropy_per_training_step"
        ] = core.agent[0].entropy_data_adversary
        data[
            "exp_" + str(self.seed) + "_adv_actor_loss_per_training_step"
        ] = core.agent[0].adversary_loss_data

        # Optimizer data
        data[
            "exp_" + str(self.seed) + "_optimizer_thermal_noise_per_training_step"
        ] = core.agent[0].optimiser_thermal_noise_data

        self.save_data(Path(self.results_dir) / "Training", **data)

        return core.agent[0], core.agent[1]

    def train_worst_adversary(self, protagonist, existing_adversary=None):
        return None

    def evaluate(self, protagonist, worst_adversary):
        data = {}
        try:
            mean_reward_vs_worst_adversary = self.evaluate_vs_internal_adversary(
                protagonist, worst_adversary, n_episodes=self.n_evaluation_episodes
            )
            data[
                "exp_" + str(self.seed) + "_mean_reward_vs_worst_adversary"
            ] = mean_reward_vs_worst_adversary
        except:
            pass
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

    def evaluate_vs_internal_adversary(self, protagonist, adversary, n_episodes):
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

        core.agent[0].set_delta(0)
        core.evaluate(n_episodes=n_episodes, render=self.bool_render)
        #core.agent[0].reset_delta()

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
            core = self.provide_core(
                "multi-agent", agents, mdp, callback_step=callbacks
            )

            return core

        core = setup()

        core.agent[0].set_delta(0)
        core.evaluate(n_episodes=n_episodes, render=self.bool_render)
        core.agent[0].reset_delta()

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
        core.agent[0].set_delta(0)

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

        core.agent[0].reset_delta()
        return mean_reward_per_metric_value
