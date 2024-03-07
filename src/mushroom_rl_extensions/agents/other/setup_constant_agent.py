from mushroom_rl_extensions.agents.abstract_setup import AbstractSetup
from mushroom_rl_extensions.core.multi_player_agent import MultiPlayerAgent
from mushroom_rl_extensions.policy.constant import ConstantPolicy


class SetupConstantAgent(AbstractSetup):
    """
    Creates an agent that outputs a constant action of equal magnitude
    across all dimensions (default magnitude=0)
    """

    @classmethod
    def provide_policy(cls, action_space_shape, constant_value):
        policy = ConstantPolicy(action_space_shape, constant_value=constant_value)
        return policy

    @classmethod
    def provide_agent(cls, mdp_info, idx_agent, **kwargs):
        constant_value = kwargs["constant_value"] if "constant_value" in kwargs else 0.0
        try:
            policy = cls.provide_policy(
                mdp_info.action_space[idx_agent].shape, constant_value=constant_value
            )
        except:
            policy = cls.provide_policy(
                mdp_info.action_space[0].shape, constant_value=constant_value
            )
        agent = MultiPlayerAgent(mdp_info, policy, idx_agent)

        return agent
