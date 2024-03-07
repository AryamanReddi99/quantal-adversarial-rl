import numpy as np
from experiments.mujoco_experiments.qarl_experiment import QARLExperiment
from experiments.mujoco_experiments.qarl_homogeneous_experiment import QARLHomogeneousExperiment
from experiments.mujoco_experiments.qarl_linear_experiment import QARLLinearExperiment
from experiments.mujoco_experiments.qarl_point_experiment import QARLPointExperiment
from experiments.mujoco_experiments.qarl_single_experiment import QARLSingleExperiment
from experiments.mujoco_experiments.baseline_experiment import BaselineExperiment
from experiments.mujoco_experiments.rarl_experiment import RARLExperiment
from experiments.mujoco_experiments.force_experiment import ForceExperiment
from experiments.mujoco_experiments.mas_experiment import MASExperiment
from experiments.mujoco_experiments.sgld_experiment import SGLDExperiment
from torch import manual_seed


def experiment(algorithm: str = "", **kwargs):
    seed = kwargs["seed"]
    np.random.seed(seed)
    manual_seed(seed)

    # Mujoco Experiment
    if algorithm == "baseline":
        experiment = BaselineExperiment(**kwargs)
    elif algorithm == "rarl":
        experiment = RARLExperiment(**kwargs)
    elif algorithm == "qarl":
        experiment = QARLExperiment(**kwargs)
    elif algorithm == "qarl_homogeneous":
        experiment = QARLHomogeneousExperiment(**kwargs)
    elif algorithm == "qarl_linear":
        experiment = QARLLinearExperiment(**kwargs)
    elif algorithm == "qarl_point":
        experiment = QARLPointExperiment(**kwargs)
    elif algorithm == "qarl_single":
        experiment = QARLSingleExperiment(**kwargs)
    elif algorithm == "mas":
        experiment = MASExperiment(**kwargs)
    elif algorithm == "sgld":
        experiment = SGLDExperiment(**kwargs)
    elif algorithm == "force":
        experiment = ForceExperiment(**kwargs)
    else:
        raise ValueError("Unknown algorithm provided!")

    protagonist, adversary = experiment.train_protagonist()
    worst_adversary = experiment.train_worst_adversary(protagonist, adversary)
    experiment.evaluate(protagonist, worst_adversary)
