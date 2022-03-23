# note: the doctsring code below within
# """ is converted to a restructuredText
# .rst file by sphinx to automatically
# generate the api's documentation
#
# docstring style used: Google style
"""
    module

    Copyright 2022 by Steeve Laquitaine, GNU license 
"""

import numpy as np


def is_all_in(x: set, y: set):
    """check if all x are in y

    Args:
        x (set): _description_
        y (set): _description_

    Returns:
        _type_: _description_
    """
    return len(x - y) == 0


def is_empty(x: np.array):
    return len(x) == 0


def sub2ind(array_shape, rows, cols):
    return rows * array_shape[1] + cols

