# note: the doctsring code below within
# """ is converted to a restructuredText
# .rst file by sphinx to automatically
# generate the api's documentation
#
# docstring style used: Google style
"""
    Visualization of circular statistics

    see:
        Hurliman et al, 2002,VR
        Stocker&Simoncelli,2006,NN
        Girshick&Simoncelli,2011,NN
        Chalk&Series,2012,JoV

    Copyright 2022 by Steeve Laquitaine, GNU license 
"""


import numpy as np
from bsfit.nodes.cirpy.data import VonMises, VonMisesMixture
from matplotlib import pyplot as plt


def plot_von_mises(v_x: np.ndarray, v_k: float):
    """plot von Mises

    Args:
        v_x (np.ndarray): support space (degree)
        v_k (float): von mises concentration (a.u.)
    
    Usage:
        .. code-block:: python

            from matplotlib import pyplot as plt
            plot_von_mises(np.arange(0, 360, 1), 10.7)
            plt.show()
    """
    # calculate prior density
    prior = VonMises(p=True).get(
        v_x, np.array([180]), [v_k]
    )

    # plot
    plt.fill_between(v_x, prior.squeeze())


def plot_von_mises_mixture(
    v_x: np.ndarray, v_u: np.ndarray, v_k: float,
):
    """plot mixture of von Mises

    Args:
        v_x (np.ndarray): support space (degree)
        v_u (np.ndarray): von mises' mean (degree)
        v_k (float): von mises concentration (a.u.)
    
    Usage: 
        .. code-block:: python

            prior = plot_von_mises_mixture(
            v_x = np.arange(0,360,1),
            v_u = np.array([90, 180, 270, 360]),
            v_k = [v_k],
            0.25
            )
    """

    # calculate prior density
    prior = VonMisesMixture(p=True).get(
        v_x, v_u, [v_k], 1 / len(v_u),
    )

    # plot
    plt.fill_between(v_x, prior.squeeze())

