# note: the doctsring code below within
# """ is converted to a restructuredText
# .rst file by sphinx to automatically
# generate the api's documentation
#
# docstring style used: Google style
"""
    Useful functions for circular statistics

    Copyright 2022 by Steeve Laquitaine, GNU license 
"""
import numpy as np
from numpy import pi


def get_deg_to_rad(deg: np.array, signed: bool):
    """convert angles in degree to radian

    Args:
        deg (np.array): angles in degree
        signed (bool): True (signed) or False (unsigned)

    Usage:
        .. code-block:: python

            import numpy as np
            from bsfit.nodes.cirpy.utils import get_deg_to_rad
            radians = get_deg_to_rad(np.array([0, 90, 180, 270], True)

            Out: array([ 0., 1.57079633, 3.14159265, -1.57079633])

    Returns:
        np.ndarray: angles in radian
    """
    # get unsigned radians (1:2*pi)
    rad = (deg / 360) * 2 * pi

    # get signed radians(-pi:pi)
    if signed:
        rad[deg > 180] = (deg[deg > 180] - 360) * (
            2 * pi / 360
        )
    return rad


def get_rad_to_deg(rad: np.ndarray):
    """convert angles in radian to degree

    Args:
        rad (np.ndarray): angles in radian
    
    Usage:
        .. code-block:: python

            import numpy as np
            from bsfit.nodes.cirpy.utils import get_rad_to_deg
            degree = get_rad_to_deg(np.array([0., 1.57079633, 3.14159265, -1.57079633]))

    Returns:
        (np.ndarray): angles in degree
    """
    # when radians are between 0:2*pi
    deg = (rad / (2 * pi)) * 360

    # when degrees are > 360, substract 360
    while sum(deg > 360):
        deg = deg - (deg > 360) * 360

    # when < - 360 degree, add 360
    while sum(deg < -360):
        deg = deg + (deg < -360) * 360

    # when radians are signed between -pi:pi.
    deg[deg < 0] = deg[deg < 0] + 360

    # replace 0 by 360
    deg[deg == 0] = 360
    return deg


def get_circ_conv(X_1: np.ndarray, X_2: np.ndarray):
    """convolve circular data

    Args:
        X_1 (np.ndarray): a column vector or a matrix        
        X_2 (np.ndarray): a column vector or a matrix
    
    Usage:
        .. code-block:: python
            
            import numpy as np
            from bsfit.nodes.cirpy.utils import get_circ_conv
            impulse = np.zeros([10,1])
            impulse[5] = 1
            convolved = get_circ_conv(np.random.rand(10,1), impulse)

            Out: 

    Returns:
        (np.array): convolved matrix

    Notes:
        Convolution is applied column-wise between columns i 
        of X_1 and i of X_2 The probability that value i in 
        vector 2 would be combined with at least one value from 
        vector 1 vector 1 and 2 are col vectors (vertical) 
    """
    convolved = []
    for col in range(X_1.shape[1]):
        convolved.append(
            np.fft.ifft(
                np.fft.fft(X_1[:, col])
                * np.fft.fft(X_2[:, col])
            ).real
        )
    return np.array(convolved).T


def get_cartesian_to_deg(
    x: np.ndarray, y: np.ndarray, signed: bool
) -> np.ndarray:
    """convert cartesian coordinates to 
    angles in degree

    Args:
        x (np.ndarray): x coordinate
        y (np.ndarray): y coordinate
        signed (boolean): True (signed) or False (unsigned)

    Usage:
        .. code-block:: python

            import numpy as np
            from bsfit.nodes.cirpy.utils import get_cartesian_to_deg
            x = np.array([1, 0, -1, 0])
            y = np.array([0, 1, 0, -1])
            degree = get_cartesian_to_deg(x,y,False)

            # Out: array([  0.,  90., 180., 270.])

    Returns:
        np.ndarray: angles in degree
    """
    # convert to radian (ignoring divide by 0 warning)
    with np.errstate(divide="ignore"):
        degree = np.arctan(y / x)

    # convert to degree and adjust based
    # on quadrant
    for ix in range(len(x)):
        if (x[ix] >= 0) and (y[ix] >= 0):
            degree[ix] = degree[ix] * 180 / np.pi
        elif (x[ix] == 0) and (y[ix] == 0):
            degree[ix] = 0
        elif x[ix] < 0:
            degree[ix] = degree[ix] * 180 / np.pi + 180
        elif (x[ix] >= 0) and (y[ix] < 0):
            degree[ix] = degree[ix] * 180 / np.pi + 360

    # if needed, convert signed to unsigned
    if not signed:
        degree[degree < 0] = degree[degree < 0] + 360
    return degree


def get_polar_to_cartesian(
    angle: np.ndarray, radius: float, type: str
) -> dict:
    """convert angle in degree or radian to cartesian coordinates

    Args:
        angle (np.ndarray): angles in degree or radian
        radius (float): radius
        type (str): "polar" or "radian"

    Usage:
        .. code-block:: python

            import numpy as np
            from bsfit.nodes.cirpy.utils import get_polar_to_cartesian
            degree = np.array([0, 90, 180, 270])
            cartesian = get_polar_to_cartesian(degree, 1, "polar")
            cartesian.keys()
            
            # Out: dict_keys(['deg', 'rad', 'cart'])
            
            cartesian["cart"]
            
            # Out: array([[ 1.,  0.],
            #            [ 0.,  1.],
            #            [-1.,  0.],
            #            [-0., -1.]])

    Returns:
        dict: _description_
    """
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
    """calculate circular data statistics

    Args:
        angle (np.ndarray): angles in degree or cartesian coordinates
        proba (np.ndarray): each angle's probability of occurrence 
        type (str): "polar" or "cartesian"

    Usage:
        .. code-block:: python

            import numpy as np
            from bsfit.nodes.cirpy.utils import get_circ_weighted_mean_std
            degree = np.array([358, 0, 2, 88, 90, 92])
            proba = np.array([1, 1, 1, 1, 1, 1])/6
            output = get_circ_weighted_mean_std(degree, proba, "polar")
            output.keys()

            # Out: dict_keys(['coord_all', 'deg_all', 'coord_mean', 'deg_mean', 
            #               'deg_all_for_std', 'deg_mean_for_std', 'deg_var', 
            #               'deg_std', 'deg_sem'])

            output["deg_mean"]

            # Out: array([45.])

            output["deg_std"]

            # array([45.02961988])

    Returns:
        (dict): angle summary statistics (mean, std, var, sem)
    
    Raises:
        ValueError: type is not "polar" or "cartesian"
    """

    angle = angle.copy()

    # if polar, convert to cartesian
    if type == "polar":
        radius = 1
        coord = get_polar_to_cartesian(
            angle, radius=radius, type="polar"
        )
    elif type == "cartesian":
        coord = angle
    else:
        raise ValueError(
            """ "type" can either be "polar" or "cartesian" value """
        )

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
        data["coord_mean"][0],
        data["coord_mean"][1],
        signed=False,
    )

    # calculate std
    # ..............
    n_data = len(data["deg_all"])
    data["deg_all_for_std"] = data["deg_all"]
    data["deg_mean_for_std"] = np.tile(
        data["deg_mean"], n_data
    )

    # apply corrections
    # when 0 <= mean <= 180
    if data["deg_mean"] + 180 <= 360:
        for ix in range(n_data):
            if (
                data["deg_all"][ix]
                >= data["deg_mean"] + 180
            ):
                data["deg_all_for_std"][ix] = (
                    data["deg_all"][ix] - 360
                )
    else:
        # when 180 <= mean <= 360
        for ix in range(n_data):
            if (
                data["deg_all"][ix]
                <= data["deg_mean"] - 180
            ):
                data["deg_mean_for_std"][ix] = (
                    data["deg_mean"] - 360
                )

    # calculate variance, standard deviation and
    # standard error to the mean
    data["deg_var"] = np.array(
        [
            sum(
                proba
                * (
                    data["deg_all_for_std"]
                    - data["deg_mean_for_std"]
                )
                ** 2
            )
        ]
    )
    data["deg_std"] = np.sqrt(data["deg_var"])
    data["deg_sem"] = data["deg_std"] / np.sqrt(n_data)
    return data


def get_signed_angle(
    origin: np.ndarray, destination: np.ndarray, type: str
):
    """get the signed angle difference between origin and destination angles

    Args:
        origin (np.ndarray): origin angle
        destination (np.ndarray): destination angle
        type (str): angle type ("polar", "radian", "cartesian")

    Usage:
        .. code-block:: python

            angle = get_signed_angle(90, 45, 'polar')
            
            # Out: array([45.])
            
            angle = get_signed_angle(90, 45, 'radian')

            # Out: array([58.3103779])

            origin = np.array([[0, 1]])
            destination = np.array([[1, 0]])
            angle = get_signed_angle(origin, destination, "cartesian")
            
            # Out: array([90.])

    Returns:
        (np.ndarray): signed angle differences
    """

    # convert to cartesian coordinates
    if type == "polar" or type == "radian":
        origin_dict = get_polar_to_cartesian(
            origin, radius=1, type=type
        )
        destination_dict = get_polar_to_cartesian(
            destination, radius=1, type=type
        )
    elif type == "cartesian":
        origin_dict = dict()
        destination_dict = dict()
        origin_dict["cart"] = origin
        destination_dict["cart"] = destination

    # get coordinates
    xV1 = origin_dict["cart"][:, 0]
    yV1 = origin_dict["cart"][:, 1]
    xV2 = destination_dict["cart"][:, 0]
    yV2 = destination_dict["cart"][:, 1]

    # Calculate the angle separating the
    # two vectors in degrees
    angle = -(180 / np.pi) * np.arctan2(
        xV1 * yV2 - yV1 * xV2, xV1 * xV2 + yV1 * yV2
    )
    return angle
