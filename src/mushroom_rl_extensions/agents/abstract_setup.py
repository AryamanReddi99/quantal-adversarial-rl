from abc import ABC, abstractmethod


class AbstractSetup(ABC):
    """
    Abstract class which can be extended to set up a specific agent
    """
    def __new__(cls, mdp_info, idx_agent, **kwargs):
        agent = cls.provide_agent(mdp_info, idx_agent, **kwargs)
        return agent

    @classmethod
    @abstractmethod
    def provide_agent(cls, mdp_info, idx_agent, **kwargs):
        raise NotImplementedError
