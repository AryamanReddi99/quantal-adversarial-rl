from copy import deepcopy
from itertools import chain

import numpy as np
import torch
import torch.autograd.functional as func
import torch.optim as optim
from mushroom_rl.algorithms.actor_critic.deep_actor_critic import DeepAC
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.policy import Policy
from mushroom_rl.utils.parameters import to_parameter
from mushroom_rl.utils.replay_memory import ReplayMemory
from mushroom_rl.utils.torch import to_float_tensor
from mushroom_rl_extensions.algorithms.actor_critic.sac import SAC


class FixedTempSAC(SAC):
    """
    Modified SAC agent that uses fixed temperature values
    instead of generic SAC temperature for actor and critic update
    """

    def __init__(self, log_alpha, **kwargs):
        super().__init__(**kwargs)
        self._log_alpha = torch.tensor(log_alpha, dtype=torch.float32)

        if self.policy.use_cuda:
            self._log_alpha = self._log_alpha.cuda().requires_grad_()
        else:
            self._log_alpha.requires_grad_()

    def _update_alpha(self, log_prob):
        return
