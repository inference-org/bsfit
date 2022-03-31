import numpy as np
from bsfit.nodes.cirpy.data import VonMises
from matplotlib import pyplot as plt


def plot_von_mises(support_space: np.ndarray, k_vm: float):

    # calculate prior density
    prior = VonMises(p=True).get(
        support_space, np.array([180]), [k_vm]
    )

    # plot
    plt.fill_between(support_space, prior.squeeze())

