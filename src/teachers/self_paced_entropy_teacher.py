import os
import pickle
import time

import torch
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, Bounds

from distributions.torch_distributions import GaussianTorchDistribution
from mushroom_rl.utils.torch import to_float_tensor


G_TOL = 1e-4        # Tolerance for termination by the norm of the Lagrangian gradient
X_TOL = 1e-6        # Tolerance for termination by the change of the independent variable


class SelfPacedEntropyTeacher:
    """
    Klink et al.: A Probabilistic Interpretation of Self-Paced Learning with Applications to Reinforcement Learning
    Implementation based on: https://github.com/psclklnk/spdl
    """
    def __init__(self, initial_mean, initial_std, target_mean, target_std, max_kl, performance_lower_bound,
                 std_lower_bound, target_kl_threshold, entropy_bounds):
        self.entropy_dist = GaussianTorchDistribution(initial_mean, initial_std, use_cuda=False, dtype=torch.float64)
        self.target_dist = GaussianTorchDistribution(target_mean, target_std, use_cuda=False, dtype=torch.float64)

        self.max_kl = max_kl
        self.perf_lb = performance_lower_bound
        self.above_perf_lb = False
        self.std_lb = std_lower_bound
        self.target_kl_threshold = target_kl_threshold
        self.entropy_bounds = entropy_bounds

        self.iteration = 0

        self._logger = None

    def compute_entropy_kl(self, old_entropy_dist):
        return torch.distributions.kl.kl_divergence(old_entropy_dist.distribution_t, self.entropy_dist.distribution_t)

    def compute_target_entropy_kl(self):
        kl_div = torch.distributions.kl.kl_divergence(self.entropy_dist.distribution_t,
                                                      self.target_dist.distribution_t).detach()
        kl_div = kl_div.cpu().numpy()
        return kl_div

    def compute_expected_performance(self, dist, entropy_t, old_entropy_log_prob_t, entropy_value_t):
        entropy_ratio_t = torch.exp(dist.log_pdf_t(entropy_t) - old_entropy_log_prob_t)
        return torch.mean(entropy_ratio_t * entropy_value_t)

    def update_entropy_distribution(self, entropies, values):
        self.iteration += 1

        old_entropy_dist = GaussianTorchDistribution.from_weights(self.entropy_dist.get_weights(),
                                                                  dtype=torch.float64)

        entropies_t = to_float_tensor(entropies, use_cuda=False)
        old_entropy_log_prob_t = old_entropy_dist.log_pdf_t(entropies_t).detach()

        # Values of initial states after the policy update
        entropy_value_t = to_float_tensor(values, use_cuda=False)

        # Define the KL constraint
        def kl_constraint_func(x):
            dist = GaussianTorchDistribution.from_weights(x, dtype=torch.float64)
            kl_div = torch.distributions.kl.kl_divergence(old_entropy_dist.distribution_t, dist.distribution_t)
            return kl_div.detach().cpu().numpy()

        def kl_constraint_grad_func(x):
            dist = GaussianTorchDistribution.from_weights(x, dtype=torch.float64)
            kl_div = torch.distributions.kl.kl_divergence(old_entropy_dist.distribution_t, dist.distribution_t)
            mu_grad, std_grad = torch.autograd.grad(kl_div, dist.parameters())
            return np.array([mu_grad.detach().cpu().numpy(), std_grad.detach().cpu().numpy()])

        kl_constraint = NonlinearConstraint(kl_constraint_func, -np.inf, self.max_kl,
                                            jac=kl_constraint_grad_func, keep_feasible=True)

        # Define the performance constraint
        def perf_constraint_func(x):
            dist = GaussianTorchDistribution.from_weights(x, dtype=torch.float64)
            perf = self.compute_expected_performance(dist, entropies_t, old_entropy_log_prob_t, entropy_value_t)
            return perf.detach().cpu().numpy()

        def perf_constraint_grad_func(x):
            dist = GaussianTorchDistribution.from_weights(x, dtype=torch.float64)
            perf = self.compute_expected_performance(dist, entropies_t, old_entropy_log_prob_t, entropy_value_t)
            mu_grad, std_grad = torch.autograd.grad(perf, dist.parameters())
            return np.array([mu_grad.detach().cpu().numpy(), std_grad.detach().cpu().numpy()])

        perf_constraint = NonlinearConstraint(perf_constraint_func, self.perf_lb, np.inf,
                                              jac=perf_constraint_grad_func, keep_feasible=True)

        if self.compute_target_entropy_kl() > self.target_kl_threshold:
            # Define the std constraint as bounds
            cones = np.ones_like(self.entropy_dist.get_weights())
            lb = -np.inf * cones.copy()
            lb[1] = self.std_lb
            ub = np.inf * cones.copy()
            bounds = Bounds(lb, ub, keep_feasible=True)

            # If the bounds are active, clip the standard deviation to be in bounds (because we may re-introduce
            # bounds after they have previously been removed)
            x0 = np.clip(self.entropy_dist.get_weights().copy(), lb, ub)
        else:
            x0 = self.entropy_dist.get_weights().copy()
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
                    dist = GaussianTorchDistribution.from_weights(x, dtype=torch.float64)
                    kl_div = torch.distributions.kl.kl_divergence(dist.distribution_t, self.target_dist.distribution_t)
                    mu_grad, std_grad = torch.autograd.grad(kl_div, dist.parameters())

                    return kl_div.detach().cpu().numpy(), \
                           np.array([mu_grad.detach().cpu().numpy(), std_grad.detach().cpu().numpy()]).astype(np.float64)

                res = minimize(objective, x0, method="trust-constr", jac=True, bounds=bounds,
                               constraints=constraints, options={"gtol": G_TOL, "xtol": X_TOL})

            # Only optimize the context distribution if the performance threshold has not yet been exceeded even once
            elif not self.above_perf_lb:
                print("Optimizing performance")
                constraints = [kl_constraint]

                # Define the objective plus Jacobian
                def objective(x):
                    dist = GaussianTorchDistribution.from_weights(x, dtype=torch.float64)
                    perf = self.compute_expected_performance(dist, entropies_t, old_entropy_log_prob_t, entropy_value_t)
                    mu_grad, std_grad = torch.autograd.grad(perf, dist.parameters())

                    return -perf.detach().cpu().numpy(), \
                           -np.array([mu_grad.detach().cpu().numpy(), std_grad.detach().cpu().numpy()]).astype(np.float64)

                res = minimize(objective, x0, method="trust-constr", jac=True, bounds=bounds,
                               constraints=constraints, options={"gtol": G_TOL, "xtol": X_TOL})
            else:
                res = None
        except Exception as e:
            os.makedirs("optimization_errors", exist_ok=True)
            with open(os.path.join("optimization_errors", "error_" + str(time.time())), "wb") as f:
                pickle.dump((self.entropy_dist.get_weights(), entropies, values), f)
            print("Exception occurred during optimization! Storing state and keeping old values!")
            # raise e
            res = None

        if res is not None and res.success:
            self.entropy_dist.set_weights(res.x)
        elif res is not None:
            # If it was not a success, but the objective value was improved and the bounds are still valid, we still
            # use the result
            old_f = objective(self.entropy_dist.get_weights())[0]
            cons_ok = True
            for constraint in constraints:
                cons_ok = cons_ok and constraint.lb <= constraint.fun(res.x) <= constraint.ub

            std_ok = bounds is None or (np.all(bounds.lb <= res.x) and np.all(res.x <= bounds.ub))
            if cons_ok and std_ok and res.fun < old_f:
                self.entropy_dist.set_weights(res.x)
            else:
                print(
                    "Warning! Context optimization unsuccessful - will keep old values. Message: %s" % res.message)
        # Logging
        new_weights = self.entropy_dist.get_weights()
        perf = perf_constraint_func(x0)
        self._log_info(perf, new_weights[0], new_weights[1])

    def sample(self):
        sample = self.entropy_dist.sample().detach().cpu().numpy()
        sample = np.clip(sample, self.entropy_bounds[0], self.entropy_bounds[1])
        return sample

    def set_logger(self, logger):
        self._logger = logger

    def _log_info(self, performance, mu, std):
        if self._logger:
            msg = "Iteration {}: \t Exp. performance: {} \t Updated mu: {} \t Updated std: {}".format(
                self.iteration, performance, mu, std)

            self._logger.info(msg)
