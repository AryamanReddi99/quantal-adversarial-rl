from mushroom_rl.utils.parameters import Parameter


class LinearBoundedParameter(Parameter):
    """
    This class implements a linearly changing parameter according to the number
    of times it has been used. Use start_gradient, end_gradient, and n to define
    the unbounded line across the entire expected n range

    """

    def __init__(
        self, start_gradient, end_gradient, n, min_value=None, max_value=None, size=(1,)
    ):
        self._coeff = (end_gradient - start_gradient) / n

        super().__init__(start_gradient, min_value, max_value, size)

        self._add_save_attr(_coeff="primitive")

    def _compute(self, *idx, **kwargs):
        return self._coeff * self._n_updates[idx] + self._initial_value
