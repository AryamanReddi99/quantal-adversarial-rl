from abc import ABC

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions import Gamma

import numpy as np

from mushroom_rl.utils.torch import to_float_tensor

"""
Based on implementations from 
mushroomRL (https://github.com/MushroomRL/mushroom-rl/tree/dev/mushroom_rl/distributions)
and self-paced deep RL (https://github.com/psclklnk/spdl/blob/master/deep_sprl/util/gaussian_torch_distribution.py)
"""


class AbstractDistribution(object):
    def sample(self):
        """
        Draw a sample from the distribution.
        Returns:
            A random vector sampled from the distribution.
        """
        raise NotImplementedError

    def log_pdf(self, x):
        """
        Compute the logarithm of the probability density function in the
        specified point
        Args:
            x (np.ndarray): the point where the log pdf is calculated
        Returns:
            The value of the log pdf in the specified point.
        """
        raise NotImplementedError

    def __call__(self, x):
        """
        Compute the probability density function in the specified point
        Args:
            x (np.ndarray): the point where the pdf is calculated
        Returns:
            The value of the pdf in the specified point.
        """
        return np.exp(self.log_pdf(x))


class TorchDistribution(AbstractDistribution, ABC):
    def __init__(self, use_cuda):
        """
        Constructor.
        Args:
            use_cuda (bool): whether to use cuda or not.
        """
        self._use_cuda = use_cuda

    def entropy(self):
        """
        Compute the entropy of the policy.
        Returns:
            The value of the entropy of the policy.
        """

        return self.entropy_t().detach().cpu().numpy()

    def entropy_t(self):
        """
        Compute the entropy of the policy.
        Returns:
            The tensor value of the entropy of the policy.
        """
        raise NotImplementedError

    def mean(self):
        """
        Compute the mean of the policy.
        Returns:
            The value of the mean of the policy.
        """
        return self.mean_t().detach().cpu().numpy()

    def mean_t(self):
        """
        Compute the mean of the policy.
        Returns:
            The tensor value of the mean of the policy.
        """
        raise NotImplementedError

    def log_pdf(self, x):
        x = to_float_tensor(x, self.use_cuda)
        return self.log_pdf_t(x).detach().cpu().numpy()

    def log_pdf_t(self, x):
        """
        Compute the logarithm of the probability density function in the
        specified point
        Args:
            x (torch.Tensor): the point where the log pdf is calculated
        Returns:
            The value of the log pdf in the specified point.
        """
        raise NotImplementedError

    def set_weights(self, weights):
        """
        Setter.
        Args:
            weights (np.ndarray): the vector of the new weights to be used by the distribution
        """
        raise NotImplementedError

    def get_weights(self):
        """
        Getter.
        Returns:
             The current policy weights.
        """
        raise NotImplementedError

    def parameters(self):
        """
        Returns the trainable distribution parameters, as expected by torch optimizers.
        Returns:
            List of parameters to be optimized.
        """
        raise NotImplementedError

    def reset(self):
        pass

    @property
    def use_cuda(self):
        """
        True if the policy is using cuda_tensors.
        """
        return self._use_cuda


class GaussianTorchDistribution(TorchDistribution):
    def __init__(self, mu, std, use_cuda, dtype):
        super().__init__(use_cuda)

        self._mu = nn.Parameter(torch.as_tensor(mu, dtype=dtype), requires_grad=True)
        self._std = nn.Parameter(torch.as_tensor(std, dtype=dtype), requires_grad=True)

        self.distribution_t = Normal(self._mu, self._std)

    def entropy_t(self):
        return self.distribution_t.entropy()

    def mean_t(self):
        return self.distribution_t.mean

    def log_pdf_t(self, x):
        return self.distribution_t.log_prob(x)

    def sample(self, *args, **kwargs):
        return self.distribution_t.rsample(*args, **kwargs)

    @staticmethod
    def from_weights(weights, use_cuda=False, dtype=torch.float32):
        mu = weights[0]
        std = weights[1]

        return GaussianTorchDistribution(mu, std, use_cuda=use_cuda, dtype=dtype)

    def set_weights(self, weights):
        if self.use_cuda:
            mu_tensor = torch.tensor([weights[0]]).type(self._mu.data.dtype).cuda()
            std_tensor = torch.tensor([weights[1]]).type(self._std.data.dtype).cuda()
        else:
            mu_tensor = torch.tensor([weights[0]]).type(self._mu.data.dtype)
            std_tensor = torch.tensor([weights[1]]).type(self._std.data.dtype)

        self._mu.data = mu_tensor
        self._std.data = std_tensor

        self.distribution_t = Normal(self._mu, self._std)

    def get_weights(self):
        mu_weight = self._mu.data.detach().cpu().numpy()
        std_weight = self._std.data.detach().cpu().numpy()

        return np.concatenate([mu_weight, std_weight])

    def parameters(self):
        return [self._mu, self._std]


class GammaTorchDistribution(TorchDistribution):
    def __init__(self, conc, rate, use_cuda, dtype):
        super().__init__(use_cuda)

        self._conc = nn.Parameter(
            torch.as_tensor(conc, dtype=dtype), requires_grad=True
        )
        self._rate = nn.Parameter(
            torch.as_tensor(rate, dtype=dtype), requires_grad=True
        )

        self.distribution_t = Gamma(self._conc, self._rate)

    def entropy_t(self):
        return self.distribution_t.entropy()

    def mean_t(self):
        return self.distribution_t.mean

    def log_pdf_t(self, x):
        return self.distribution_t.log_prob(x)

    def sample(self, *args, **kwargs):
        return self.distribution_t.rsample(*args, **kwargs)

    @staticmethod
    def from_weights(weights, use_cuda=False, dtype=torch.float32):
        conc = weights[0]
        rate = weights[1]

        return GammaTorchDistribution(conc, rate, use_cuda=use_cuda, dtype=dtype)

    def set_weights(self, weights):
        if self.use_cuda:
            conc_tensor = torch.tensor([weights[0]]).type(self._conc.data.dtype).cuda()
            rate_tensor = torch.tensor([weights[1]]).type(self._rate.data.dtype).cuda()
        else:
            conc_tensor = torch.tensor([weights[0]]).type(self._conc.data.dtype)
            rate_tensor = torch.tensor([weights[1]]).type(self._rate.data.dtype)

        self._conc.data = conc_tensor
        self._rate.data = rate_tensor

        self.distribution_t = Gamma(self._conc, self._rate)

    def get_weights(self):
        conc_weight = self._conc.data.detach().cpu().numpy()
        rate_weight = self._rate.data.detach().cpu().numpy()

        return np.concatenate(np.array([conc_weight, rate_weight]))

    def parameters(self):
        return [self._conc, self._rate]
