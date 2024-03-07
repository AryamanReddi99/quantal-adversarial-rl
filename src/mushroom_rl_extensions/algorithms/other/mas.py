import torch
import torch.autograd.functional as func
import numpy as np
from numpy import linalg as LA
from mushroom_rl.core.agent import Agent
from mushroom_rl.policy.policy import Policy


class MASPolicy(Policy):
    """
    Policy that acts as MAS adversarial agent in Curriculum Adversarial Training (CAT) algorithm.
    Based on https://ieeexplore.ieee.org/document/9892908

    budget: allowed actions-space perturbation budget of adversarial policy
    lr: gradient descent learning rate of adversarial policy
    descent_steps_limit: maximum gradient descent steps taken by adversarial policy
    epsilon: convergence threshold for subsequent gradient descent steps
    lp_norm: norm of MAS budget. Experiment were conducted with lp_norm=2 as this was found to improve the robustness
    """
    def __init__(self, budget, lr, descent_steps_limit, epsilon, lp_norm):
        self.budget = budget
        self.lr = lr
        self.descent_steps_limit = descent_steps_limit
        self.epsilon = epsilon
        self.lp_norm = lp_norm

        self.prot_distribution = None
        self.prot_action = None

        assert budget >= 0

        super().__init__()

        # Data collection
        self.mas_action_norm_data = []

    def draw_action(self, state):

        grad_a = self.compute_policy_gradient(self.prot_action)
        mas_action = self.prot_action - (self.lr * grad_a)

        grad_a = self.compute_policy_gradient(mas_action)
        mas_action_new = mas_action - (self.lr * grad_a)

        # debug
        actions = []
        actions.extend([self.prot_action, mas_action, mas_action_new])

        counter = 0
        while (
            np.absolute(mas_action - mas_action_new) > self.epsilon
        ).any() and counter < self.descent_steps_limit:
            mas_action = mas_action_new
            grad_a = self.compute_policy_gradient(mas_action)
            mas_action_new = mas_action - (self.lr * grad_a)
            counter += 1

            # debug
            actions.append(mas_action_new)

        delta = mas_action_new - self.prot_action
        if self.lp_norm == 2:
            proj_spatial_delta = self.l2_spatial_project(delta)
        elif self.lp_norm == 1:
            proj_spatial_delta = self.l1_spatial_project(delta)

        if state[2] < 1:
            pass

        # Data collection
        action_norm = np.linalg.norm(proj_spatial_delta)
        self.mas_action_norm_data.append(action_norm)

        return proj_spatial_delta

    def l2_spatial_project(self, x):
        norm = LA.norm(x, 2)
        if norm <= self.budget:
            delta = x
        else:
            delta = (x / norm) * self.budget
        return delta

    def l1_spatial_project(self, x):
        """Compute the Euclidean projection on a L1-ball
        Solves the optimisation problem (using the algorithm from [1]):
            min_w 0.5 * || w - x ||_2^2 , s.t. || w ||_1 <= self.budget
        Parameters
        ----------
        x: (n,) numpy array,
        n-dimensional vector to project
        Returns
        -------
        w: (n,) numpy array,
        Euclidean projection of x on the L1-ball of radius self.budget
        Notes
        -----
        Solves the problem by a reduction to the positive simplex case
        See also
        --------
        euclidean_proj_simplex
        """

        def euclidean_project_simplex(x):
            """Compute the Euclidean projection on a positive simplex
            Solves the optimisation problem (using the algorithm from [1]):
                min_w 0.5 * || w - x ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0
            Parameters
            ----------
            x: (n,) numpy array,
            n-dimensional vector to project
            Returns
            -------
            w: (n,) numpy array,
            Euclidean projection of v on the simplex
            Notes
            -----
            The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
            Better alternatives exist for high-dimensional sparse vectors (cf. [1])
            However, this implementation still easily scales to millions of dimensions.
            References
            ----------
            [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
                John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
                International Conference on Machine Learning (ICML 2008)
                http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
            """
            (n,) = x.shape  # will raise ValueError if v is not 1-D
            # check if we are already on the simplex
            if x.sum() == self.budget and np.alltrue(x >= 0):
                # best projection: itself!
                return x
            # get the array of cumulative sums of a sorted (decreasing) copy of v
            u = np.sort(x)[::-1]
            cssv = np.cumsum(u)
            # get the number of > 0 components of the optimal solution
            rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - self.budget))[0][-1]
            # compute the Lagrange multiplier associated to the simplex constraint
            # theta = float(cssv[rho] - s) / rho
            theta = float(cssv[rho] - self.budget) / (rho + 1)

            # compute the projection by thresholding v using theta
            w = (x - theta).clip(min=0)
            return w

        (n,) = x.shape  # will raise ValueError if v is not 1-D
        # compute the vector of absolute values
        u = np.abs(x)
        # check if x is already a solution
        if u.sum() <= self.budget:
            # L1-norm is <= self.budget
            return x
        # x is not already a solution: optimum lies on the boundary (norm == self.budget)
        # project *u* on the simplex
        w = euclidean_project_simplex(u)
        # compute the solution to the original problem on v
        w *= np.sign(x)
        return w

    def compute_policy_gradient(self, action: np.ndarray):
        action_t = torch.tensor(action)
        grad_a = (
            func.jacobian(lambda x: self.prot_distribution.log_prob(x).exp(), action_t)
            .detach()
            .cpu()
            .numpy()
        )
        return np.array([grad_a[i][i] for i in range(len(action_t))])


class MAS(Agent):
    """
    MAS Adversary
    https://arxiv.org/pdf/1909.02583.pdf
    """

    def __init__(
        self,
        mdp_info,
        idx_agent,
        budget: float = 1.0,
        lr: float = 3,
        descent_steps_limit: int = 25,
        epsilon: float = 0.001,
        lp_norm: int = 2,
        **kwargs
    ):
        self._idx_agent = idx_agent
        self._budget = budget

        policy = MASPolicy(
            budget=budget,
            lr=lr,
            descent_steps_limit=descent_steps_limit,
            epsilon=epsilon,
            lp_norm=lp_norm,
        )

        super().__init__(mdp_info=mdp_info, policy=policy)

        self._add_save_attr(_idx_agent="primitive", _budget="primitive")

    def set_prot_policy_distribution(self, distribution):
        """
        Set's the protagonist policy distribution for a given state
        """
        self.policy.prot_distribution = distribution

    def set_prot_action(self, action):
        """
        Set's the protagonist's action for a given state
        """
        self.policy.prot_action = action

    def set_budget(self, budget):
        self.policy.budget = budget
