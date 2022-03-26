import numpy as np
from bsfit.nodes.circpy import get_signed_angle
from matplotlib import pyplot as plt


def plot_mean(
    data_mean: np.ndarray,
    data_std: np.ndarray,
    prediction_mean: np.ndarray,
    prediction_std: np.ndarray,
    condition: np.ndarray,
    prior_mode: float,
    centering: bool,
):
    """plot data and prediction mean and std
    for three conditions (x-axis, colors and panels)

    Args:
        data_mean (np.ndarray): data mean by condition
        data_std (np.ndarray): data std by condition
        prediction_mean (np.ndarray): prediction mean by condition
        prediction_std (np.ndarray): prediction std by condition
        condition (np.ndarray): associated conditions
        prior_mode (float): the mode of the prior
        centering (bool): center x-axis or not

    Returns:
        _type_: _description_
    """
    # get condition levels
    levels_1 = np.unique(condition[:, 0])
    levels_2 = np.unique(condition[:, 1])
    levels_3 = np.unique(condition[:, 2])

    # set x_tick
    x_tick_centered = get_signed_angle(
        levels_1, prior_mode, "polar"
    )
    x_tick_centered[x_tick_centered == -180] = 180
    i_sort = np.argsort(x_tick_centered)
    x_tick_centered = x_tick_centered[i_sort]
    y_tick_centered = x_tick_centered

    # set colors
    levels_2_color = [
        [0.5, 0, 0],
        [1, 0.2, 0],
        [1, 0.6, 0],
        [0.75, 0.75, 0],
    ]
    levels_2_color_prediction = [
        [0.2, 0, 0],
        [0.97, 0.2, 0],
        [0.8, 0.4, 0],
        [0.3, 0.3, 0],
    ]
    levels_2_color_er_prediction = [
        [0.2, 0.5, 0.5],
        [0.97, 0.7, 0.5],
        [0.8, 0.9, 0, 5],
        [0.3, 0.9, 0.5],
    ]
    # loop over conditions and plot data
    # and prediction stats
    for level2_ix in range(len(levels_2)):

        # set condition 2 in column panels
        plt.subplot(1, len(levels_2), level2_ix + 1)

        # set condition 3 within panels
        for level1_ix in range(len(levels_1)):

            # find condition's instances
            loc_lev1 = (
                condition[:, 0] == levels_1[level1_ix]
            )
            loc_lev2 = (
                condition[:, 1] == levels_2[level2_ix]
            )
            loc_condition = loc_lev2 & loc_lev1

            # center to prior mode
            x_centered = condition[:, 2][loc_condition]
            if centering:
                x_centered = np.round(
                    get_signed_angle(
                        x_centered, prior_mode, "polar",
                    )
                )

            x_centered[x_centered == -180] = 180

            # make 2-D array
            x_centered = x_centered[:, None]

            # sort data stats
            y_data_centered = data_mean[loc_condition]
            y_data_std_centered = data_std[loc_condition]

            # sort prediction stats
            y_centered = prediction_mean[loc_condition]
            y_std_centered = prediction_std[loc_condition]

            # sort all
            i_sort = np.argsort(x_centered.squeeze())
            x_centered = x_centered[i_sort]
            y_centered = y_centered[i_sort]
            y_std_centered = y_std_centered[i_sort]
            y_data_centered = y_data_centered[i_sort]
            y_data_std_centered = y_data_std_centered[
                i_sort
            ]

            # To plot estimates mean against circular stimulus
            # feature on a linear space, the raw stimulus feature and
            # estimates mean are normalized to vectorial angles from
            # the prior mode and x and y axes are centered at zero
            # (normalized prior mode) via a circular shift. Rotation
            # angles were then labelled according to their raw values
            # on the circle (e.g., 0, is labelled 225). A mean estimate
            # of 33 degree was calculated for 55 degree which is very far
            # from stimulus feature on the linear space but actually close
            # to stimulus feature on the circular space. We got rid of
            # this visual artifact by expressing both 55 and 33 as the
            # counterclockwise distance to prior mode (e.g., for a prior
            # mode 225 55 becomes 190 instead of 170 and 33 becomes 168).
            # Note that the maximum vectorial angle is >180.
            if (level2_ix == 3) & (not 180 in x_centered):

                # move point at -170? distance to prior at 190? (positive
                # side) and convert values at x=-170? to positive distance
                # relative to prior to improve visualization
                posNeg170 = x_centered == -170
                x_centered[posNeg170] = (
                    prior_mode - 170 + 360 - prior_mode
                )
                x_centered[x_centered == 180] = -180
                y_centered[posNeg170] = (
                    prior_mode
                    - np.abs[y_centered[posNeg170]]
                    + 360
                    - prior_mode
                )
                y_centered[y_centered == 180] = -180
                y_data_centered[posNeg170] = (
                    prior_mode
                    - abs[y_data_centered[posNeg170]]
                    + 360
                    - prior_mode
                )

                # sort x-axis
                i_sort = np.argsort(x_centered)
                x_centered = x_centered[i_sort]

                # sort y-axis
                y_data_centered = y_data_centered[i_sort]
                y_centered = y_centered[i_sort]

                # set ticks
                x_tick_centered = x_centered
                y_tick_centered = x_centered

            # plot data
            plt.errorbar(
                x_centered.squeeze(),
                y_data_centered.squeeze(),
                yerr=y_data_std_centered.squeeze(),
                marker="o",
                markersize=5,
                color=levels_2_color[level2_ix],
                ecolor=levels_2_color[level2_ix],
            )

            # plot predictions
            # mean
            plt.plot(
                x_centered,
                y_centered,
                y_std_centered,
                color=levels_2_color_prediction[level2_ix],
            )

            # std
            plt.fill_between(
                x_centered.squeeze(),
                y_centered.squeeze()
                - y_std_centered.squeeze(),
                y_centered.squeeze()
                + y_std_centered.squeeze(),
                color=levels_2_color_er_prediction[
                    level2_ix
                ],
            )
    plt.show()
    return None

