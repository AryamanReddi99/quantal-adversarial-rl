import numpy as np

import torch
import torch.optim as optim
import torch.autograd.functional as func

from mushroom_rl_extensions.algorithms.actor_critic.sac import SAC, SACPolicy
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.spaces import Box

from copy import deepcopy
from itertools import chain


class SAC_LD(SAC):
    """
    SAC with SGLD adversary.
    "Robust Reinforcement Learning via
    Adversarial training with Langevin Dynamics"
    Kamalaruban et al.. 2020.
    """

    def __init__(self, delta, initial_thermal_noise, mdp_info, **kwargs):
        # Modify mdp_info action space
        mdp_info_ld = deepcopy(mdp_info)
        new_adversary_max_force_low = mdp_info_ld.action_space[1].low[0]
        new_adversary_max_force_high = mdp_info_ld.action_space[1].high[0]
        adv_action_space_shape = np.ones(mdp_info_ld.action_space[0].shape)
        adv_action_space_low = adv_action_space_shape * new_adversary_max_force_low
        adv_action_space_high = adv_action_space_shape * new_adversary_max_force_high
        adv_action_space = Box(adv_action_space_low, adv_action_space_high)
        mdp_info_ld.action_space[1] = adv_action_space

        super().__init__(mdp_info=mdp_info_ld, **kwargs)
        self.original_delta = delta
        self.delta = delta

        self.initial_thermal_noise = initial_thermal_noise

        actor_mu_params = kwargs["actor_mu_params"]
        actor_sigma_params = kwargs["actor_sigma_params"]
        actor_optimizer = kwargs["actor_optimizer"]
        lr_alpha_adversary = kwargs["lr_alpha"]
        log_std_min = kwargs["log_std_min"]
        log_std_max = kwargs["log_std_max"]

        # Adversary approximators
        adversary_mu_params = deepcopy(actor_mu_params)
        adversary_sigma_params = deepcopy(actor_sigma_params)
        adversary_mu_approximator = Regressor(TorchApproximator, **adversary_mu_params)
        adversary_sigma_approximator = Regressor(
            TorchApproximator, **adversary_sigma_params
        )

        adversary_policy = SACPolicy(
            adversary_mu_approximator,
            adversary_sigma_approximator,
            self.mdp_info.action_space[1].low,
            self.mdp_info.action_space[1].high,
            log_std_min,
            log_std_max,
        )

        self._log_alpha_adversary = torch.tensor(
            -5.30, dtype=torch.float32
        )  # alpha = 0.005

        if self.policy.use_cuda:
            self._log_alpha_adversary = (
                self._log_alpha_adversary.cuda().requires_grad_()
            )
        else:
            self._log_alpha_adversary.requires_grad_()

        self._alpha_adversary_optim = optim.Adam(
            [self._log_alpha_adversary], lr=lr_alpha_adversary
        )

        adversary_policy_parameters = chain(
            adversary_mu_approximator.model.network.parameters(),
            adversary_sigma_approximator.model.network.parameters(),
        )

        # Adversary optimizer and parameters
        adversary_optimizer = deepcopy(actor_optimizer)
        if adversary_optimizer is not None:
            if adversary_policy_parameters is not None and not isinstance(
                adversary_policy_parameters, list
            ):
                adversary_policy_parameters = list(adversary_policy_parameters)
            self._adversary_policy_parameters = adversary_policy_parameters

            self._adversary_optimizer = adversary_optimizer["class"](
                adversary_policy_parameters, **adversary_optimizer["params"]
            )
            self._adversary_clipping = None
            if "clipping" in adversary_optimizer:
                self._adversary_clipping = adversary_optimizer["clipping"]["method"]
                self._adversary_clipping_params = adversary_optimizer["clipping"][
                    "params"
                ]

        self.adversary_policy = adversary_policy

        self._add_save_attr(
            delta="primitive",
            initial_thermal_noise="primitive",
            _log_alpha_adversary="torch",
            _alpha_adversary_optim="torch",
            _adversary_optimizer="torch",
            _adversary_clipping="torch",
            _adversary_clipping_params="pickle",
            adversary_policy="mushroom",
        )

        # Data collection
        self.temperature_data_adversary = []
        self.entropy_data_adversary = []
        self.adversary_loss_data = []
        self.optimiser_thermal_noise_data = []

    def draw_action(self, state):
        mu = self.policy.draw_action(state)
        mu_clip = np.clip(
            mu, self.mdp_info.action_space[0].low, self.mdp_info.action_space[0].high
        ) * (1 - self.delta)
        net_action = mu_clip

        try:
            adv_mu = self.adversary_policy.draw_action(state)
            adv_mu_clip = (
                np.clip(
                    adv_mu,
                    self.mdp_info.action_space[1].low,
                    self.mdp_info.action_space[1].high,
                )
                * self.delta
            )
            net_action += adv_mu_clip
        except:
            pass

        return net_action

    def fit(self, dataset):
        own_dataset = self.split_dataset(dataset)
        self._replay_memory.add(own_dataset)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, _ = self._replay_memory.get(
                self._batch_size()
            )

            # Actor updates
            if self._replay_memory.size > self._warmup_transitions():
                # Actor update
                (
                    action_new_actor,
                    log_prob_actor,
                ) = self.policy.compute_action_and_log_prob_t(state)
                (
                    action_new_adversary,
                    log_prob_adversary,
                ) = self.adversary_policy.compute_action_and_log_prob_t(state)
                action_new = (
                    self.delta * action_new_adversary
                    + (1 - self.delta) * action_new_actor
                )

                q_0 = self._critic_approximator(
                    state, action_new, output_tensor=True, idx=0
                )
                q_1 = self._critic_approximator(
                    state, action_new, output_tensor=True, idx=1
                )
                q = torch.min(q_0, q_1)
                actor_loss = self._actor_loss(q, log_prob_actor)
                self._optimize_actor_parameters(actor_loss)

                # Adversary update
                (
                    action_new_actor,
                    log_prob_actor,
                ) = self.policy.compute_action_and_log_prob_t(state)
                (
                    action_new_adversary,
                    log_prob_adversary,
                ) = self.adversary_policy.compute_action_and_log_prob_t(state)
                action_new = (
                    self.delta * action_new_adversary
                    + (1 - self.delta) * action_new_actor
                )

                q_0 = self._critic_approximator(
                    state, action_new, output_tensor=True, idx=0
                )
                q_1 = self._critic_approximator(
                    state, action_new, output_tensor=True, idx=1
                )
                q = torch.min(q_0, q_1)
                adversary_loss = self._adversary_loss(q, log_prob_adversary)
                self._optimize_adversary_parameters(adversary_loss)

                # Update alphas
                self._update_alpha(log_prob_actor.detach())
                self._update_alpha_adversary(log_prob_adversary.detach())

            # Critic update
            q_next = self._next_q(next_state, absorbing)
            q = reward + self.mdp_info.gamma * q_next

            self._critic_approximator.fit(state, action, q, **self._critic_fit_params)

            self._update_target(
                self._critic_approximator, self._target_critic_approximator
            )

            # Optimizer noise updates
            noise = self.initial_thermal_noise * (
                (1 - 5e-5) ** (self._fit_iteration / 2)
            )
            self._update_thermal_noise(noise)

            self._fit_iteration += 1

        # Collect data
        if self._replay_memory.initialized:
            if self._replay_memory.size > self._warmup_transitions():
                self.temperature_data.append(self._alpha_np)
                self.entropy_data.append(self.policy.entropy(state))
                self.actor_loss_data.append(actor_loss.detach().cpu().numpy())

                self.temperature_data_adversary.append(self._alpha_adversary_np)
                self.entropy_data_adversary.append(self.adversary_policy.entropy(state))
                self.adversary_loss_data.append(adversary_loss.detach().cpu().numpy())

                self.optimiser_thermal_noise_data.append(noise)

            critic_loss = list()
            for i in range(self._critic_approximator.__len__()):
                model_i = self._critic_approximator.__getitem__(i)
                if hasattr(model_i, "loss_fit"):
                    m_loss = model_i.loss_fit
                    if hasattr(m_loss, "squeeze"):
                        m_loss = m_loss.squeeze()
                    critic_loss.append(m_loss)
            self.critic_loss_data.append(critic_loss)

    def _actor_loss(self, q, log_prob_actor):
        return (self._alpha * log_prob_actor - q).mean()

    def _adversary_loss(self, q, log_prob_adversary):
        return (self._alpha_adversary * log_prob_adversary + q).mean()

    def _optimize_actor_parameters(self, loss):
        self._optimizer.zero_grad()
        loss.backward()
        self._clip_gradient()
        self._optimizer.step()

    def _optimize_adversary_parameters(self, adversary_loss):
        self._adversary_optimizer.zero_grad()
        adversary_loss.backward()
        self._adversary_clip_gradient()
        self._adversary_optimizer.step()

    def _adversary_clip_gradient(self):
        if self._adversary_clipping:
            self._adversary_clipping(
                self._adversary_policy_parameters, **self._adversary_clipping_params
            )

    def _update_alpha_adversary(self, log_prob_adversary):
        alpha_loss = -(
            self._log_alpha_adversary * (log_prob_adversary + self._target_entropy)
        ).mean()
        self._alpha_adversary_optim.zero_grad()
        alpha_loss.backward()
        self._alpha_adversary_optim.step()

    def _next_q(self, next_state, absorbing):
        (
            action_new_actor,
            log_prob_actor,
        ) = self.policy.compute_action_and_log_prob(next_state)

        (
            action_new_adversary,
            log_prob_adversary,
        ) = self.policy.compute_action_and_log_prob(next_state)

        action_new = (
            self.delta * action_new_adversary + (1 - self.delta) * action_new_actor
        )

        q = (
            self._target_critic_approximator.predict(
                next_state, action_new, prediction="min"
            )
            - self._alpha_np * log_prob_actor
        )
        q *= 1 - absorbing

        return q

    @property
    def _alpha_adversary(self):
        return self._log_alpha_adversary.exp()

    @property
    def _alpha_adversary_np(self):
        return self._alpha_adversary.detach().cpu().numpy()

    def _update_thermal_noise(self, noise):
        self._optimizer.param_groups[0]["noise"] = noise
        self._adversary_optimizer.param_groups[0]["noise"] = noise

    def set_delta(self, delta):
        self.delta = delta

    def reset_delta(self):
        self.delta = self.original_delta
