import numpy as np
import torch

from mushroom_rl_extensions.algorithms.actor_critic.sac import SAC
from mushroom_rl.utils.torch import to_float_tensor


class AdaptiveTempSAC(SAC):
    """
    Modified SAC agent that uses temperature values stored in state instead of generic SAC temperature
    for actor and critic update
    """
    def __init__(
        self,
        mdp_info,
        idx_agent,
        actor_mu_params,
        actor_sigma_params,
        actor_optimizer,
        critic_params,
        batch_size,
        initial_replay_size,
        max_replay_size,
        warmup_transitions,
        tau,
        lr_alpha,
        log_std_min=-20,
        log_std_max=2,
        target_entropy=None,
        critic_fit_params=None,
    ):
        super().__init__(
            mdp_info=mdp_info,
            idx_agent=idx_agent,
            actor_mu_params=actor_mu_params,
            actor_sigma_params=actor_sigma_params,
            actor_optimizer=actor_optimizer,
            critic_params=critic_params,
            batch_size=batch_size,
            initial_replay_size=initial_replay_size,
            max_replay_size=max_replay_size,
            warmup_transitions=warmup_transitions,
            tau=tau,
            lr_alpha=lr_alpha,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            target_entropy=target_entropy,
            critic_fit_params=critic_fit_params,
        )

        # overwriting the initial scalar alpha tensor by alpha tensor of batch_size dimension
        self._log_alpha = torch.tensor(np.zeros(batch_size), dtype=torch.float32)

    def fit(self, dataset):
        own_dataset = self.split_dataset(dataset)
        self._replay_memory.add(own_dataset)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, _ = self._replay_memory.get(
                self._batch_size()
            )
            # alpha values are getting dictated by the self-paced teacher
            self._update_alpha(state[:, -1])

            # Actor update
            if self._replay_memory.size > self._warmup_transitions():
                action_new, log_prob = self.policy.compute_action_and_log_prob_t(state)
                loss = self._loss(state, action_new, log_prob)
                self._optimize_actor_parameters(loss)

            # Critic update
            q_next = self._next_q(next_state, absorbing)
            q = reward + self.mdp_info.gamma * q_next

            self._critic_approximator.fit(state, action, q, **self._critic_fit_params)

            self._update_target(
                self._critic_approximator, self._target_critic_approximator
            )

        self._fit_iteration += 1

        # Collect data
        if (
            self._replay_memory.initialized
            and self._replay_memory.size > self._warmup_transitions()
        ):
            self.temperature_data.append(np.mean(self._alpha_np))
            self.entropy_data.append(self.policy.entropy(state))
            self.actor_loss_data.append(loss.detach().cpu().numpy())

            critic_loss = list()
            for i in range(self._critic_approximator.__len__()):
                model_i = self._critic_approximator.__getitem__(i)
                if hasattr(model_i, "loss_fit"):
                    m_loss = model_i.loss_fit
                    if hasattr(m_loss, "squeeze"):
                        m_loss = m_loss.squeeze()
                    critic_loss.append(m_loss)
            self.critic_loss_data.append(critic_loss)

        elif self._replay_memory.initialized:
            critic_loss = list()
            for i in range(self._critic_approximator.__len__()):
                model_i = self._critic_approximator.__getitem__(i)
                if hasattr(model_i, "loss_fit"):
                    m_loss = model_i.loss_fit
                    if hasattr(m_loss, "squeeze"):
                        m_loss = m_loss.squeeze()
                    critic_loss.append(m_loss)
            self.critic_loss_data.append(critic_loss)

        # Print fit information
        # if self._fit_iteration % 5000 == 0:
        #     if self._replay_memory.initialized and self._replay_memory.size > self._warmup_transitions():
        #         self._log_info(loss, state)

    def _update_alpha(self, temperatures):
        if self.policy.use_cuda:
            temp_t = to_float_tensor(temperatures, use_cuda=True)
        else:
            temp_t = to_float_tensor(temperatures, use_cuda=False)
        self._log_alpha = torch.log(temp_t)

    def _log_info(self, loss, states):
        if self._logger:
            actor_loss = loss.detach().cpu().numpy()

            critic_loss = list()
            for i in range(self._critic_approximator.__len__()):
                model_i = self._critic_approximator.__getitem__(i)
                if hasattr(model_i, "loss_fit"):
                    m_loss = model_i.loss_fit
                    if hasattr(m_loss, "squeeze"):
                        m_loss = m_loss.squeeze()
                    critic_loss.append(m_loss)
            critic_loss = np.array(critic_loss).squeeze()

            batch_alpha = np.mean(self._alpha_np)
            batch_mean_entropy_per_dimension = self.policy.entropy(states)

            msg = (
                "Fit Iteration {}: \t actor loss: {} \t critic loss: {} \t batch alpha: {} \t "
                "batch mean entropy per action dimension: {} ".format(
                    self._fit_iteration,
                    actor_loss,
                    critic_loss,
                    batch_alpha,
                    batch_mean_entropy_per_dimension,
                )
            )

            self._logger.info(msg)
