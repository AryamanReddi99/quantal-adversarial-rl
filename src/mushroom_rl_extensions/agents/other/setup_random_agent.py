from mushroom_rl_extensions.agents.abstract_setup import AbstractSetup
from mushroom_rl_extensions.core.multi_player_agent import MultiPlayerAgent
from mushroom_rl_extensions.policy.random import RandomContinuousPolicy


class SetupRandomAgent(AbstractSetup):
    """
    Creates an agent that outputs a random action within the specific action space
    """

    @classmethod
    def provide_policy(cls, action_space):
        policy = RandomContinuousPolicy(action_space)
        return policy

    @classmethod
    def provide_agent(cls, mdp_info, idx_agent, **kwargs):
        try:
            policy = cls.provide_policy(mdp_info.action_space[idx_agent])
        except:
            policy = cls.provide_policy(mdp_info.action_space[0])
        agent = MultiPlayerAgent(mdp_info, policy, idx_agent)

        return agent
