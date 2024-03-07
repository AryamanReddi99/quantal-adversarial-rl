from mushroom_rl_extensions.agents.deep_rl.setup_adaptive_temp_sac_agent import (
    SetupAdaptiveTempSACAgent,
)
from mushroom_rl_extensions.agents.deep_rl.setup_fixed_temp_sac_agent import (
    SetupFixedTempSACAgent,
)
from mushroom_rl_extensions.agents.deep_rl.setup_sac_agent import SetupSACAgent
from mushroom_rl_extensions.agents.deep_rl.setup_sgld_sac_agent import SetupSGLDSACAgent
from mushroom_rl_extensions.agents.other.setup_constant_agent import SetupConstantAgent
from mushroom_rl_extensions.agents.other.setup_mas_agent import SetupMASAgent
from mushroom_rl_extensions.agents.other.setup_random_agent import SetupRandomAgent


class SetupAgent:
    """
    Creates an agent using the appropriate setup object
    """

    def __new__(cls, agent, mdp_info, idx_agent, **kwargs):
        if agent == "sac":
            agent = SetupSACAgent(mdp_info, idx_agent, **kwargs)
        elif agent == "adaptive_temp_sac":
            agent = SetupAdaptiveTempSACAgent(mdp_info, idx_agent, **kwargs)
        elif agent == "fixed_temp_sac":
            agent = SetupFixedTempSACAgent(mdp_info, idx_agent, **kwargs)
        elif agent == "sac_ld":
            agent = SetupSGLDSACAgent(mdp_info, idx_agent, **kwargs)
        elif agent == "mas":
            agent = SetupMASAgent(mdp_info, idx_agent, **kwargs)
        elif agent == "constant":
            agent = SetupConstantAgent(mdp_info, idx_agent, **kwargs)
        elif agent == "random":
            agent = SetupRandomAgent(mdp_info, idx_agent, **kwargs)
        else:
            raise ValueError("Unknown agent name provided!")

        return agent
