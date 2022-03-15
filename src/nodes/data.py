import numpy as np
from numpy import cos, exp, pi
from scipy.special import iv


class VonMises:
    def __init__(self, p: bool):
        self.p = p

    def get(self, v_x, v_u, v_k):
        """Create von Mises functions or probability
            distributions

        Args:
            v_x (np.array): support space
            v_u (list): mean
            v_k (list): concentration

        Returns:
            _type_: _description_
        """

        # radians
        x_rad = self._get_deg_to_rad(v_x, True)
        u_rad = self._get_deg_to_rad(v_u, True)

        # when k is the same and means are different
        if sum(np.array(v_k) - v_k[0]) == 0:
            vmises = self._get_same_k_different_means(
                x_rad, u_rad, v_x, v_u, v_k
            )
        return vmises

    def _get_same_k_different_means(
        self, x_rad, u_rad, v_x, v_u, v_k
    ):
        # When von mises with different mean u1,u2,u3 but with same k are input
        # We can get von Mises with mean u2,u3,etc...simply by rotating the von
        # mises with mean u1 by u2-u1, u3-u1 etc...
        # When we don't do that we get slightly different von mises with different
        # peakvalue due to numerical instability caused by cosine and exponential
        # functions.
        # case all k are same
        if sum(np.array(v_k) - v_k[0]) == 0:

            # when mean is not in x
            if not self._is_all_in(set(v_u), set(v_x)):
                print(
                    """(get_vonMises) The mean "u" 
                    is not in space "x".
                    Choose "u" in "x"."""
                )
                # when there are missing means
                if any(np.isnan(v_u)):
                    print(
                        """(get_vonMises) The mean 
                    is nan ..."""
                    )
            else:
                # when k -> +inf
                # make the von mises delta functions
                if v_k[0] > 1e300:
                    vmises = np.zeros(
                        (len(x_rad), len(v_u))
                    )
                    vmises[x_rad == u_rad[0], 0] = 1

                    # get other von mises by circular
                    # translation of the first
                    rot_start = np.where(v_x == v_u[0])[0]
                    for i in range(1, len(v_u)):
                        rot_end = np.where(v_x == v_u[i])[0]
                        rotation = rot_end - rot_start
                        vmises[:, i] = np.roll(
                            vmises[:, 0], rotation
                        )
                else:
                    # initialize matrix of von mises
                    # for each mean
                    vmises = (
                        np.zeros((len(x_rad), len(v_u)))
                        * np.nan
                    )

                    # when k is not +inf
                    amp = 1
                    v_k = v_k[0]
                    bessel_order = 0
                    scaling = 2 * pi * iv(bessel_order, v_k)
                    vm_fun = exp(
                        v_k * cos(amp * (x_rad - u_rad[0]))
                        - v_k
                    )
                    vmises[:, 0] = vm_fun / scaling

                    # create other von mises by shifting
                    # the first circularly
                    rot_start = np.where(v_x == v_u[0])[0]
                    for i in range(1, len(v_u)):
                        rot_end = np.where(v_x == v_u[i])[0]
                        rotation = rot_end - rot_start
                        vmises[:, i] = np.roll(
                            vmises[:, 0], rotation
                        )

                    # normalize to probabilities
                    if self.p:
                        vmises = vmises / sum(vmises[:, 0])
            return vmises

    def _get_deg_to_rad(self, deg: float, signed: bool):
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

    def _is_all_in(self, x: set, y: set):
        """check if all x are in y

        Args:
            x (set): _description_
            y (set): _description_

        Returns:
            _type_: _description_
        """
        return len(x - y) == 0

