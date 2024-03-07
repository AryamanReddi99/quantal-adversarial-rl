import os
import pickle
import time

import torch
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, Bounds

from distributions.torch_distributions import GammaTorchDistribution
from mushroom_rl.utils.torch import to_float_tensor

G_TOL = 1e-4  # Tolerance for termination by the norm of the Lagrangian gradient
X_TOL = 1e-6  # Tolerance for termination by the change of the independent variable


class SelfPacedGammaTemperatureTeacher:
    """
    Klink et al.: A Probabilistic Interpretation of Self-Paced Learning with Applications to Reinforcement Learning
    Implementation based on: https://github.com/psclklnk/spdl
    """

    def __init__(
        self,
        initial_conc,
        initial_rate,
        target_conc,
        target_rate,
        max_kl,
        performance_lower_bound,
        conc_lower_bound,
        rate_lower_bound,
        target_kl_threshold,
        temperature_bounds,
    ):
        self.temp_dist = GammaTorchDistribution(
            initial_conc, initial_rate, use_cuda=False, dtype=torch.float64
        )
        self.target_dist = GammaTorchDistribution(
            target_conc, target_rate, use_cuda=False, dtype=torch.float64
        )

        self.max_kl = max_kl
        self.perf_lb = performance_lower_bound
        self.above_perf_lb = False
        self.conc_lb = conc_lower_bound
        self.rate_lb = rate_lower_bound
        self.target_kl_threshold = target_kl_threshold
        self.temp_bounds = temperature_bounds

        self.iteration = 0

        self._logger = None

        # Data collection
        self.conc_temperature_data = [
            self.temp_dist.parameters()[0].data.detach().cpu().numpy()[0]
        ]
        self.rate_temperature_data = [
            self.temp_dist.parameters()[1].data.detach().cpu().numpy()[0]
        ]

    def compute_temp_kl(self, old_temp_dist):
        return torch.distributions.kl.kl_divergence(
            self.temp_dist.distribution_t, old_temp_dist.distribution_t
        )

    def compute_target_temp_kl(self):
        kl_div = torch.distributions.kl.kl_divergence(
            self.temp_dist.distribution_t, self.target_dist.distribution_t
        ).detach()
        kl_div = kl_div.cpu().numpy()
        return kl_div

    def compute_expected_performance(
        self, dist, temp_t, old_temp_log_prob_t, temp_value_t
    ):
        temp_ratio_t = torch.exp(dist.log_pdf_t(temp_t) - old_temp_log_prob_t)
        return torch.mean(temp_ratio_t * temp_value_t)

    def update_temperature_distribution(self, temperatures, values, training_iteration):
        # self.iteration and training_iteration are different
        self.iteration += 1

        old_temp_dist = GammaTorchDistribution.from_weights(
            self.temp_dist.get_weights(), dtype=torch.float64
        )

        temps_t = to_float_tensor(temperatures, use_cuda=False)
        old_temp_log_prob_t = old_temp_dist.log_pdf_t(temps_t).detach()

        # Values of initial states after the policy update
        temp_value_t = to_float_tensor(values, use_cuda=False)

        # Define the KL constraint
        def kl_constraint_func(x):
            dist = GammaTorchDistribution.from_weights(x, dtype=torch.float64)
            kl_div = torch.distributions.kl.kl_divergence(
                dist.distribution_t, old_temp_dist.distribution_t
            )
            return kl_div.detach().cpu().numpy()

        def kl_constraint_grad_func(x):
            dist = GammaTorchDistribution.from_weights(x, dtype=torch.float64)
            kl_div = torch.distributions.kl.kl_divergence(
                dist.distribution_t, old_temp_dist.distribution_t
            )
            conc_grad, rate_grad = torch.autograd.grad(kl_div, dist.parameters())
            return np.array(
                [conc_grad.detach().cpu().numpy(), rate_grad.detach().cpu().numpy()]
            )

        kl_constraint = NonlinearConstraint(
            kl_constraint_func,
            -np.inf,
            self.max_kl,
            jac=kl_constraint_grad_func,
            keep_feasible=True,
        )

        # Define the performance constraint
        def perf_constraint_func(x):
            dist = GammaTorchDistribution.from_weights(x, dtype=torch.float64)
            perf = self.compute_expected_performance(
                dist, temps_t, old_temp_log_prob_t, temp_value_t
            )
            return perf.detach().cpu().numpy()

        def perf_constraint_grad_func(x):
            dist = GammaTorchDistribution.from_weights(x, dtype=torch.float64)
            perf = self.compute_expected_performance(
                dist, temps_t, old_temp_log_prob_t, temp_value_t
            )
            conc_grad, rate_grad = torch.autograd.grad(perf, dist.parameters())
            return np.array(
                [conc_grad.detach().cpu().numpy(), rate_grad.detach().cpu().numpy()]
            )

        perf_constraint = NonlinearConstraint(
            perf_constraint_func,
            self.perf_lb,
            np.inf,
            jac=perf_constraint_grad_func,
            keep_feasible=True,
        )

        if self.compute_target_temp_kl() > self.target_kl_threshold:
            # Define the conc and rate constraint as bounds
            cones = np.ones_like(self.temp_dist.get_weights())
            lb = cones.copy()
            lb[0] = self.conc_lb
            lb[1] = self.rate_lb
            ub = np.inf * cones.copy()
            bounds = Bounds(lb, ub, keep_feasible=True)

            # If the bounds are active, clip the standard deviation to be in bounds (because we may re-introduce
            # bounds after they have previously been removed)
            x0 = np.clip(self.temp_dist.get_weights().copy(), lb, ub)
        else:
            x0 = self.temp_dist.get_weights().copy()
            bounds = None

        try:
            if kl_constraint_func(x0) >= self.max_kl:
                print("Warning! KL-Bound of x0 violates constraint already")

            if perf_constraint_func(x0) >= self.perf_lb:
                print("Optimizing KL divergence")
                self.above_perf_lb = True
                constraints = [kl_constraint, perf_constraint]

                # Define the objective plus Jacobian
                def objective(x):
                    dist = GammaTorchDistribution.from_weights(x, dtype=torch.float64)
                    kl_div = torch.distributions.kl.kl_divergence(
                        dist.distribution_t, self.target_dist.distribution_t
                    )
                    conc_grad, rate_grad = torch.autograd.grad(
                        kl_div, dist.parameters()
                    )

                    return kl_div.detach().cpu().numpy(), np.array(
                        [
                            conc_grad.detach().cpu().numpy(),
                            rate_grad.detach().cpu().numpy(),
                        ]
                    ).astype(np.float64)

                res = minimize(
                    objective,
                    x0,
                    method="trust-constr",
                    jac=True,
                    bounds=bounds,
                    constraints=constraints,
                    options={"gtol": G_TOL, "xtol": X_TOL},
                )

            # Only optimize the temperature distribution if the performance threshold has not yet been exceeded even
            # once
            elif not self.above_perf_lb:
                print("Optimizing performance")
                constraints = [kl_constraint]

                # Define the objective plus Jacobian
                def objective(x):
                    dist = GammaTorchDistribution.from_weights(x, dtype=torch.float64)
                    perf = self.compute_expected_performance(
                        dist, temps_t, old_temp_log_prob_t, temp_value_t
                    )
                    conc_grad, rate_grad = torch.autograd.grad(perf, dist.parameters())

                    return -perf.detach().cpu().numpy(), -np.array(
                        [
                            conc_grad.detach().cpu().numpy(),
                            rate_grad.detach().cpu().numpy(),
                        ]
                    ).astype(np.float64)

                res = minimize(
                    objective,
                    x0,
                    method="trust-constr",
                    jac=True,
                    bounds=bounds,
                    constraints=constraints,
                    options={"gtol": G_TOL, "xtol": X_TOL},
                )
            else:
                res = None
        except Exception as e:
            os.makedirs("optimization_errors", exist_ok=True)
            with open(
                os.path.join("optimization_errors", "error_" + str(time.time())), "wb"
            ) as f:
                pickle.dump((self.temp_dist.get_weights(), temperatures, values), f)
            print(
                "Exception occurred during optimization! Storing state and keeping old values!"
            )
            # raise e
            res = None

        if res is not None and res.success:
            self.temp_dist.set_weights(res.x)
        elif res is not None:
            # If it was not a success, but the objective value was improved and the bounds are still valid, we still
            # use the result
            old_f = objective(self.temp_dist.get_weights())[0]
            cons_ok = True
            for constraint in constraints:
                cons_ok = (
                    cons_ok and constraint.lb <= constraint.fun(res.x) <= constraint.ub
                )

            rate_ok = bounds is None or (
                np.all(bounds.lb <= res.x) and np.all(res.x <= bounds.ub)
            )
            if cons_ok and rate_ok and res.fun < old_f:
                self.temp_dist.set_weights(res.x)
            else:
                print(
                    "Warning! Context optimization unsuccessful - will keep old values. Message: %s"
                    % res.message
                )

        # Collect data
        # 30.000 steps until temperature update but 3 steps per SAC update
        # UPDATE: now done on a per-update basis
        self.conc_temperature_data.extend(
            [self.conc_temperature_data[-1]]
            * (training_iteration - len(self.conc_temperature_data))
        )
        self.conc_temperature_data.append(
            self.temp_dist.parameters()[0].data.detach().cpu().numpy()[0]
        )
        self.rate_temperature_data.extend(
            [self.rate_temperature_data[-1]]
            * (training_iteration - len(self.rate_temperature_data))
        )
        self.rate_temperature_data.append(
            self.temp_dist.parameters()[1].data.detach().cpu().numpy()[0]
        )

        # Logging
        new_weights = self.temp_dist.get_weights()
        perf = perf_constraint_func(new_weights)
        self._log_info(
            perf,
            new_weights[0],
            new_weights[1],
            self.compute_temp_kl(old_temp_dist),
            self.compute_target_temp_kl(),
        )

    def sample(self):
        sample = self.temp_dist.sample().detach().cpu().numpy()
        sample = np.clip(sample, self.temp_bounds[0], self.temp_bounds[1])
        return sample

    def set_logger(self, logger):
        self._logger = logger

    def _log_info(self, performance, conc, rate, temp_kl, target_temp_kl):
        if self._logger:
            msg = "It.: {} E[J]: {} conc: {} rate: {} KL(old||new): {} KL(new||target): {}".format(
                self.iteration,
                round(float(performance), 6),
                round(conc, 6),
                round(rate, 6),
                round(float(temp_kl.detach().cpu().numpy()[0]), 6),
                round(float(target_temp_kl[0]), 6),
            )

            self._logger.info(msg)
