import os
import pickle
import time

import numpy as np

G_TOL = 1e-4  # Tolerance for termination by the norm of the Lagrangian gradient
X_TOL = 1e-6  # Tolerance for termination by the change of the independent variable


class SelfPacedPointTemperatureTeacher:
    """
    Klink et al.: A Probabilistic Interpretation of Self-Paced Learning with Applications to Reinforcement Learning
    Implementation based on: https://github.com/psclklnk/spdl
    Teacher which
    """

    def __init__(
        self,
        initial_alpha,
        target_alpha,
        max_mse,
        performance_lower_bound,
        temp_lower_bound,
        target_mse_threshold,
    ):
        self.temp = initial_alpha
        self.target_temp = target_alpha

        self.max_mse = max_mse
        self.perf_lb = performance_lower_bound
        self.above_perf_lb = False
        self.temp_lb = temp_lower_bound
        self.target_mse_threshold = target_mse_threshold

        self.iteration = 0

        self._logger = None

        # Data collection
        self.temperature_data = [self.temp]

    def compute_target_temp_mse(self):
        return (self.temp - self.target_temp) ** 2

    def update_temperature_distribution(self, temps, values, training_iteration):
        # self.iteration and training_iteration are different
        self.iteration += 1

        if self.compute_target_temp_mse() > self.target_mse_threshold:
            if np.mean(values) > self.perf_lb:
                self.temp -= np.sqrt(self.max_mse)
                self.temp = np.clip(self.temp, a_min=self.temp_lb, a_max=None)
            else:
                pass
        else:
            pass

        # Collect data
        # 30.000 steps until temperature update but 3 steps per SAC update
        # UPDATE: now done on a per-update basis
        self.temperature_data.extend(
            [self.temperature_data[-1]]
            * (training_iteration - len(self.temperature_data))
        )
        self.temperature_data.append(self.temp)

        # Logging
        self._log_info(
            np.mean(values),
        )

    def sample(self):
        return self.temp

    def set_logger(self, logger):
        self._logger = logger

    def _log_info(self, performance):
        if self._logger:
            msg = "It.: {} E[J]: {} temp: {} target_temp_mse: {}".format(
                self.iteration,
                round(float(performance), 6),
                round(float(self.temp), 6),
                round(float(self.compute_target_temp_mse()), 6),
            )

            self._logger.info(msg)
