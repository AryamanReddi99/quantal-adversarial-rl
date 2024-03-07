from mushroom_rl.core.agent import Agent


class MultiPlayerAgent(Agent):
    def __init__(self, mdp_info, policy, idx_agent, features=None):
        super().__init__(mdp_info, policy, features)

        self._idx_agent = idx_agent

        self._add_save_attr(
            _idx_agent='primitive'
        )

    def fit(self, dataset):
        """
        Fit step.

        Args:
            dataset (list): the dataset.

        """
        raise NotImplementedError('MultiPlayerAgent is an abstract class')
