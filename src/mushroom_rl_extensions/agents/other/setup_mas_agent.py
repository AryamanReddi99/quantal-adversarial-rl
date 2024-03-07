from mushroom_rl_extensions.agents.abstract_setup import AbstractSetup
from mushroom_rl_extensions.algorithms.other.mas import MAS


class SetupMASAgent(AbstractSetup):
    """
    Creates an agent that acts as the adversary for a 
    Myopic Action Space (MAS) attack strategy. Based on:
    Based on https://arxiv.org/abs/1909.02583
    """
    @classmethod
    def provide_agent(cls, mdp_info, idx_agent, **kwargs):
        agent = MAS(mdp_info, idx_agent, **kwargs)
        return agent
