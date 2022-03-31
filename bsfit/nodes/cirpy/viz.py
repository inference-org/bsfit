import numpy as np
from bsfit.nodes.cirpy.data import VonMises, VonMisesMixture
from matplotlib import pyplot as plt


def plot_von_mises(support_space: np.ndarray, k_vm: float):

    # calculate prior density
    prior = VonMises(p=True).get(
        support_space, np.array([180]), [k_vm]
    )

    # plot
    plt.fill_between(support_space, prior.squeeze())


def plot_von_mises_mixture(
    support_space: np.ndarray, v_u: np.ndarray, v_k: float,
):
    """_summary_

    Args:
        support_space (np.ndarray): _description_
        k_means (np.ndarray): _description_
        k_vm (float): _description_
    
    Usage: 
        .. code-block:: python

            prior = plot_von_mises_mixture(
            support_space=np.arange(0,360,1),
            v_u=np.array([90, 180, 270, 360]),
            v_k=[k_vm],
            0.25
            )
    """

    # calculate prior density
    prior = VonMisesMixture(p=True).get(
        support_space, v_u, [v_k], 1/len(v_u),
    )

    # plot
    plt.fill_between(support_space, prior.squeeze())

