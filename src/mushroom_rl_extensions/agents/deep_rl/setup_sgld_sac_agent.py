from mushroom_rl_extensions.agents.abstract_setup import AbstractSetup
from mushroom_rl_extensions.utils.optimizers import SGLD

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mushroom_rl_extensions.algorithms.actor_critic.sac_ld import SAC_LD


class ActorNetwork(nn.Module):
    """
    Generic actor network architecture
    """
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(ActorNetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._in = nn.Linear(n_input, n_features)
        self._h1 = nn.Linear(n_features, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._out = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._in.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._out.weight, gain=nn.init.calculate_gain("linear"))

    def forward(self, state):
        in_features = torch.squeeze(state, 1).float()

        features1 = F.relu(self._in(in_features))
        features2 = F.relu(self._h1(features1))
        features3 = F.relu(self._h2(features2))

        actions = self._out(features3)

        return actions


class CriticNetwork(nn.Module):
    """
    Generic critic network architecture
    """
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._in = nn.Linear(n_input, n_features)
        self._h1 = nn.Linear(n_features, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._out = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._in.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self._out.weight, gain=nn.init.calculate_gain("linear"))

    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)

        features1 = F.relu(self._in(state_action))
        features2 = F.relu(self._h1(features1))
        features3 = F.relu(self._h2(features2))

        q = self._out(features3)

        return torch.squeeze(q)


class SetupSGLDSACAgent(AbstractSetup):
    """
    Instantiates a multiplayer SAC agent with Langevin dynamics with the chosen parameters
    """
    INITIAL_REPLAY_SIZE = 3000
    MAX_REPLAY_SIZE = int(1e6)
    BATCH_SIZE = 256
    N_FEATURES = 256
    WARMUP_TRANSITIONS = 5000
    TAU = 0.005

    LR_ALPHA = 3e-4
    LR_ACTOR = 3e-4
    LR_CRITIC = 3e-4

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2
    TARGET_ENTROPY = None

    DELTA = 0.1
    INITIAL_THERMAL_NOISE = 1e-3

    @classmethod
    def provide_agent(cls, mdp_info, idx_agent, **kwargs):
        actor_mu_params = dict(
            network=ActorNetwork,
            n_features=cls.N_FEATURES,
            input_shape=mdp_info.observation_space.shape,
            output_shape=mdp_info.action_space[idx_agent].shape,
            use_cuda=kwargs.get("use_cuda", False),
        )

        actor_sigma_params = dict(
            network=ActorNetwork,
            n_features=cls.N_FEATURES,
            input_shape=mdp_info.observation_space.shape,
            output_shape=mdp_info.action_space[idx_agent].shape,
            use_cuda=kwargs.get("use_cuda", False),
        )

        actor_optimizer = {
            "class": SGLD,
            "params": {"lr": cls.LR_ACTOR, "noise": 1e-3, "alpha": 0.999},
        }

        critic_input_shape = (
            mdp_info.observation_space.shape[0]
            + mdp_info.action_space[idx_agent].shape[0],
        )
        critic_params = dict(
            network=CriticNetwork,
            optimizer={"class": optim.Adam, "params": {"lr": cls.LR_CRITIC}},
            loss=F.mse_loss,
            n_features=cls.N_FEATURES,
            input_shape=critic_input_shape,
            output_shape=(1,),
            use_cuda=kwargs.get("use_cuda", False),
        )

        agent = SAC_LD(
            mdp_info=mdp_info,
            idx_agent=idx_agent,
            actor_mu_params=actor_mu_params,
            actor_sigma_params=actor_sigma_params,
            actor_optimizer=actor_optimizer,
            critic_params=critic_params,
            batch_size=cls.BATCH_SIZE,
            initial_replay_size=cls.INITIAL_REPLAY_SIZE,
            max_replay_size=cls.MAX_REPLAY_SIZE,
            warmup_transitions=cls.WARMUP_TRANSITIONS,
            tau=cls.TAU,
            lr_alpha=cls.LR_ALPHA,
            log_std_min=cls.LOG_STD_MIN,
            log_std_max=cls.LOG_STD_MAX,
            target_entropy=cls.TARGET_ENTROPY,
            critic_fit_params=None,
            delta=cls.DELTA,
            initial_thermal_noise=cls.INITIAL_THERMAL_NOISE,
        )

        return agent
