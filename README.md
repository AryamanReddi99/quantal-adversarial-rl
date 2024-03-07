
# Quantal Adversarial Reinforcement Learning

## About
This paper implements the code for the paper '[Robust Adversarial Reinforcement Learning via Bounded Rationality Curricula](https://arxiv.org/abs/2311.01642)', including the 'Quantal Adversarial RL' algorithm and baselines.
  

## Installation

Use your preferred Python package manager to install the required Python libraries from requirements.txt. We recommend using a virtual environment.

```bash

pip  install  -r  requirements.txt

```

  

## Usage

All experiments can be run using the top-level script, main.py.

  

Running this script from the command line allows the user to set the parameters of the experiment they wish to run with the desired number of seeds. Possible experiment parameters include the domain, task, and algorithm.

  

All results are stored in the results folder by default.

  

To view the metrics of an experiment graphically, simply run the script 'src/evaluation/plot_results.py' in the command line with the 'results_path' argument set to the top-level folder of the desired experiment. This will plot the time-varying metrics of the environment such as mean temperature and score, as well as the final evaluation data such as robustness.

  

## Requirements

Python 3.8
