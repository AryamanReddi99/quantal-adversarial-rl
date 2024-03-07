from pathlib import Path

from mushroom_rl.core.logger.logger import Logger
from mushroom_rl_extensions.agents.create_agent import SetupAgent
from tqdm import trange

from .abstract_experiment import AbstractExperiment


class RARLExperiment(AbstractExperiment):
    """
    Runs a robust adversarial reinforcement learning training algorithm
    Based on http://proceedings.mlr.press/v70/pinto17a/pinto17a.pdf
    """

    def train_protagonist(self):
        def setup():
            mdp = self.provide_mdp()
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

            if "quadruped" in self._domain_name:
                adversary = SetupAgent(
                    self.agent,
                    mdp.info,
                    idx_agent=1,
                    use_cuda=self.use_cuda,
                    adv_n_features=64,
                )
            else:
                adversary = SetupAgent(
                    self.agent, mdp.info, idx_agent=1, use_cuda=self.use_cuda
                )
            adv_logger = Logger(
                log_name="Adversary",
                results_dir=Path(self.results_dir) / "Logging",
                log_console=True,
                seed=self.seed,
                console_log_level=30,
            )
            adversary.set_logger(adv_logger)

            agents = [protagonist, adversary]

            core = self.provide_core("multi-agent", agents, mdp)

            return core, prot_logger, adv_logger

        core, prot_logger, adv_logger = setup()

        # Train agents
        mean_reward_vs_adversary_progress = []
        mean_reward_without_adversary_progress = []
        for i in trange(self.n_total_iterations, leave=False):
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
            adv_logger.log_best_agent(core.agent[1], -mean_reward_vs_adversary)

        self.save_agents(
            core.agent,
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

        # Action data
        data[
            "exp_" + str(self.seed) + "_prot_action_norm_per_training_step"
        ] = core.action_norms[0]
        data[
            "exp_" + str(self.seed) + "_adv_action_norm_per_training_step"
        ] = core.action_norms[1]

        self.save_data(Path(self.results_dir) / "Training", **data)

        return core.agent[0], core.agent[1]
