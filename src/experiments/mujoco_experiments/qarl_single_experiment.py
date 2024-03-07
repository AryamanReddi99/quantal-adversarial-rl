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
from tqdm import trange
from utils.buffer import Buffer
from utils.state_preprocessor import StateVariableConcatenation

from .abstract_experiment import AbstractExperiment
from .qarl_experiment import QARLExperiment


class QARLSingleExperiment(QARLExperiment):
    """
    Runs an adversarial experiment that trains an agent against an adversary with a self-paced temperature curriculum (QARL)
    that only uses one temperature in each training iteration
    """

    @staticmethod
    def sample_n_temperatures(teacher, n):
        sampled_temperatures = np.zeros(n)
        temp = teacher.sample()
        for i in range(n):
            sampled_temperatures[i] = temp

        return sampled_temperatures
