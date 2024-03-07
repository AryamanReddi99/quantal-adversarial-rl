from mushroom_rl.core.agent import Agent

from mushroom_rl.utils.table import Table


class RandomAgent(Agent):
    """
    Initialize with random policy object
    """
    def __init__(self, mdp_info, policy, idx_agent):
        approximator = Table(mdp_info.action_space[idx_agent].size)
        policy.set_approximator(approximator)

        super().__init__(mdp_info, policy)

    def fit(self, dataset):
        pass
