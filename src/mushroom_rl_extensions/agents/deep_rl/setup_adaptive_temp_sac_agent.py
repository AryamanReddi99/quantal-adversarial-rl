import torch.nn.functional as F
import torch.optim as optim

from .setup_sac_agent import SetupSACAgent, ActorNetwork, CriticNetwork

from mushroom_rl_extensions.algorithms.actor_critic.adaptive_temp_sac import (
    AdaptiveTempSAC,
)


class SetupAdaptiveTempSACAgent(SetupSACAgent):
    """
    Instantiates a multiplayer adaptive temperature SAC agent used for QARL
    with the chosen parameters
    """
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

        actor_optimizer = {"class": optim.Adam, "params": {"lr": cls.LR_ACTOR}}

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

        agent = AdaptiveTempSAC(
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
        )

        return agent
