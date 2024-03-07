import numpy as np


def linear_bounded_function(a, b, x_a, x_b, x):
    """
    Function that outputs a linear function of x that takes some value
    a at x_a, b at x_b, and is flat on either side of x_a and x_b.
    """

    slope = (b - a) / (x_b - x_a)
    output_unbounded = a + slope * (x - x_a)
    output_bounded = np.clip(output_unbounded, b, a)
    return output_bounded
