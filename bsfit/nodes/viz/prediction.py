import numpy as np
from bsfit.nodes.circpy import get_signed_angle
from matplotlib import pyplot as plt


def plot_mean(
    data_mean,
    data_std,
    prediction_mean,
    prediction_std,
    condition,
    prior_mode,
):

    # get condition levels
    levels_1 = np.unique(condition[:, 0])
    levels_2 = np.unique(condition[:, 1])
    levels_3 = np.unique(condition[:, 2])

    # loop over condition and plot stats
    for level2_ix in range(len(levels_2)):
        plt.subplot(1, len(levels_2), level2_ix + 1)
        for level3_ix in range(len(levels_3)):
            loc_lev2 = (
                condition[:, 1] == levels_2[level2_ix]
            )
            loc_lev3 = (
                condition[:, 3] == levels_3[level3_ix]
            )
            loc_condition = loc_lev2 & loc_lev3

            # center to prior mode
            x_centered = np.round(
                get_signed_angle(
                    condition[:, 0][loc_condition],
                    prior_mode,
                    "polar",
                )
            )
            x_centered[x_centered == -180] = 180

    return None

