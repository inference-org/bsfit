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


def get_deg_to_rad(deg: np.array, signed: bool):
    """convert degree angles to radians

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


def get_rad_to_deg(rad: float):
    """convert radian angles to degrees

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
    """calculate circular convolutions 
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


def get_cartesian_to_deg(x, y):

    # round
    x = np.round(x, 4)
    y = np.round(y, 4)

    # convert cartesian to radian
    radian = np.arctan(y / x)

    for ix in range(len(x)):
        if (x[ix] >= 0) and (y[ix] >= 0):
            radian[ix] = radian[ix] * 180 / np.pi
        elif x[ix] < 0:
            radian[ix] = radian[ix] * 180 / np.pi + 180
        elif (x[ix] >= 0) and (y[ix] < 0):
            radian[ix] = radian[ix] * 180 / np.pi + 360
    return get_rad_to_deg(radian)


def get_polar_to_cartesian(
    angle: np.ndarray, radius: float, type: str
) -> dict:

    # convert to radian if needed
    theta = dict()
    if type == "polar":
        theta["deg"] = angle
        theta["rad"] = angle * np.pi / 180
    elif type == "radian":
        theta["deg"] = get_deg_to_rad(angle, False)
        theta["rad"] = angle

    # convert to cartesian coordinates
    x = radius * np.cos(theta["rad"])
    y = radius * np.sin(theta["rad"])

    # round to 10e-4
    x = np.round(x, 4)
    y = np.round(y, 4)

    # reshape as (N angles x 2 coord)
    theta["cart"] = np.vstack([x, y]).T
    return theta


def get_circ_weighted_mean_std(
    angle: np.ndarray, proba: np.ndarray, type: str
) -> dict:
    """calculate the circular mean and standard 
    deviation of an array of angles

    Args:
        angle (np.ndarray): _description_
        proba (np.ndarray): _description_
        type (str): _description_

    Returns:
        (dict): angle mean and std
    """
    # if polar, convert to cartesian
    if type == "polar":
        radius = 1
        coord = get_polar_to_cartesian(
            angle, radius=radius, type="polar"
        )
    else:
        coord = angle

    # store angles
    data = dict()
    data["coord_all"] = coord["cart"]
    data["deg_all"] = coord["deg"]

    # calculate mean
    # ..............
    proba_for_mean = np.tile(proba[:, None], 2)
    data["coord_mean"] = np.sum(
        proba_for_mean * data["coord_all"], 0
    )
    data["coord_mean"] = data["coord_mean"][:, None]
    data["deg_mean"] = get_cartesian_to_deg(
        data["coord_mean"][0], data["coord_mean"][1]
    )

    # calculate std
    # ..............
    n_data = len(data["deg_all"])
    data["deg_all_for_std"] = data["deg_all"]
    data["deg_mean_for_std"] = np.tile(
        data["deg_mean"], n_data
    )

    # apply corrections
    if data["deg_mean"] + 180 <= 360:
        for ix in range(n_data):
            if (
                data["deg_all"][ix]
                >= data["deg_mean"] + 180
            ):
                data["deg_all_for_std"][ix] = (
                    data["deg_all"] - 360
                )
    else:
        for ix in range(n_data):
            if (
                data["deg_all"][ix]
                <= data["deg_mean"] - 180
            ):
                data["deg_mean_for_std"][ix] = (
                    data["deg_mean"] - 360
                )

    # get variance, standard deviation and
    # standard error to the mean
    data["deg_var"] = sum(
        proba
        * (
            data["deg_all_for_std"]
            - data["deg_mean_for_std"]
        )
        ** 2
    )
    data["deg_std"] = np.sqrt(data["deg_var"])
    data["deg_sem"] = data["deg_std"] / np.sqrt(n_data)
    return data


def get_signed_angle(
    angle_1: np.ndarray, angle_2: np.ndarray, type: str
):
    """get signed angle between angle 1 and 2

    Args:
        angle_1 (np.ndarray): _description_
        angle_2 (np.ndarray): _description_
        type (str): _description_
        - "polar"
        - "radian"
        - "cartesian"

    Returns:
        (np.ndarray): angle difference between 
        angle 1 and 2
    
    Usage:
    .. code-block:: python

        angle = get_signed_angle(90, 45, 'polar')
        angle = get_signed_angle(90, 45, 'radian')
        angle = get_signed_angle([0 1], [1 0])
    """

    # convert to cartesian coordinates
    if type == "polar" or type == "radian":
        angle_1 = get_polar_to_cartesian(
            angle_1, radius=1, type=type
        )
        angle_2 = get_polar_to_cartesian(
            angle_2, radius=1, type=type
        )

    # get coordinates
    xV1 = angle_1["cart"][:, 0]
    yV1 = angle_1["cart"][:, 1]
    xV2 = angle_2["cart"][:, 0]
    yV2 = angle_2["cart"][:, 1]

    # Calculate the angle separating the
    # two vectors in degrees
    angle = -(180 / np.pi) * np.arctan2(
        xV1 * yV2 - yV1 * xV2, xV1 * xV2 + yV1 * yV2
    )
    return angle
