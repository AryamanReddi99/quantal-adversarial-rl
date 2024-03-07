from mushroom_rl.utils.parameters import Parameter
import math


class LittmanDecay(Parameter):
    """
    This class implements an exponentially changing decay according to the
    number of times it has been used. The decay gets multiplied with the initial value.
    "Markov Games as a Framework for Multi-Agent Reinforcement Learning". Michael L. Littman. 1994.

    """
    def __init__(self, n_steps, init_value=1.0, decay_factor=0.01, size=(1,)):
        self._decay = pow(10, math.log(decay_factor, 10) / n_steps)

        super().__init__(init_value, min_value=decay_factor, max_value=None, size=size)

        self._add_save_attr(_decay='primitive')

    def _compute(self, *idx, **kwargs):
        return self._initial_value * (self._decay ** self._n_updates[idx])
