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
from numpy import pi


def is_all_in(x: set, y: set):
    """check if all x are in y

    Args:
        x (set): _description_
        y (set): _description_

    Returns:
        _type_: _description_
    """
    return len(x - y) == 0


def get_deg_to_rad(deg: np.array, signed: bool):
    """convert degrees to radians

    Args:
        deg (np.array): _description_
        signed (bool): _description_

    Returns:
        np.array: _description_
    """
    # get unsigned radians (1:2*pi)
    rad = (deg / 360) * 2 * pi

    # get signed radians(-pi:pi)
    if signed:
        rad[deg > 180] = (deg[deg > 180] - 360) * (
            2 * pi / 360
        )
    return rad


def is_empty(x: np.array):
    return len(x) == 0


def get_rad_to_deg(rad: float):
    """convert angles in radians to 
    degrees

    Args:
        rad (float): angles in radians

    Returns:
        _type_: _description_
    """
    # when input radians are between 0:2*pi
    deg = (rad / (2 * pi)) * 360

    # when degrees are > 360, substract 360
    while sum(deg > 360):
        deg = deg - (deg > 360) * 360

    # when < -360 degrees, add 360
    while sum(deg < -360):
        deg = deg + (deg < -360) * 360

    # When radians are signed b/between -pi:pi.
    deg[deg < 0] = deg[deg < 0] + 360

    # ensure angles are between 1:360
    deg[deg == 0] = 360
    return deg


def get_circ_conv(X_1: np.ndarray, X_2: np.ndarray):
    """Calculate circular convolutions 
    for circular data.
    The probability that value i in vector 2 would be 
    combined with at least one value from vector 1
    vector 1 and 2 are col vectors (vertical) 
    or matrices for convolution 
    column by column. e.g., two probability densities

    Args:
        X_1 (np.ndarray): _description_
        X_2 (np.ndarray): _description_

    Raises:
        ValueError: _description_
    """
    return np.fft.ifft(
        np.fft.fft(X_1) * np.fft.fft(X_2)
    ).real


def sub2ind(array_shape, rows, cols):
    return rows * array_shape[1] + cols
