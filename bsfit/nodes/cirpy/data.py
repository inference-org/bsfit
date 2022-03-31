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
from numpy import cos, exp, pi
from scipy.special import iv

from .utils import get_deg_to_rad


class VonMises:
    """Von Mises data class
    
    von Mises are created based on the
    equation vm=exp(k.*cos(x-u))./(2*pi.*besseli(0,k)). The code works for any
    value of k (but not for +inf). The equation is adjusted because of the
    following numerical issues: when k>700, vm is NaN because besseli(0,k) and
    exp(k.*cos(x-u)) reach numerical limits. exp(k.*cos(x-u)-k) scales vm
    without changing its shape. besseli(0,k,1)) does same. The adjusted
    equation and the exact equation yield exact same results except that the
    adjusted equation works for large k (>>700).
 
    When k > 1e300 densities become delta densities. We generate delta 
    density for k>300
    """

    def __init__(self, p: bool):
        """instantiate Von Mises 

        Args:
            p (bool): 
                True/False means probabilities 
                or not
        """
        self.p = p

    def get(self, v_x: np.array, v_u: np.array, v_k: list):
        """Create von Mises functions or probability
            distributions

        Args:
            v_x (np.array): von mises' support space
            v_u (np.array): von mises' mean
            v_k (list): von mises' concentration

        Returns:
            _type_: _description_
        """

        # radians
        x_rad = get_deg_to_rad(v_x, True)
        u_rad = get_deg_to_rad(v_u, True)

        # when k is the same and means are different
        if is_unique(v_k):
            vmises = self._get_same_k_different_means(
                x_rad, u_rad, v_x, v_u, v_k
            )
        else:
            vmises = self._get_different_k_and_means(
                x_rad, u_rad, v_k
            )
        return vmises

    def _get_same_k_different_means(
        self,
        x_rad: np.array,
        u_rad: np.array,
        v_x: np.array,
        v_u: list,
        v_k: list,
    ):
        """ get von mises with the same k 
        but different means

        Args:
            x_rad (np.array):
            u_rad (np.array):
            v_x (np.array):
            v_u (list):
            v_k (list):
        
        Returns:
            (np.array): matrix of von mises

        Description:
            When von mises with different mean u1,u2,u3 but with same k are input
            We can get von Mises with mean u2,u3,etc...simply by rotating the von
            mises with mean u1 by u2-u1, u3-u1 etc...
            When we don't do that we get slightly different von mises with different
            peakvalue due to numerical instability caused by cosine and exponential
            functions.
            case all k are same
            when mean is not in x    
        """

        # case the mean is not in the support space
        if not self._is_all_in(set(v_u), set(v_x)):
            raise Exception(
                """(get_vonMises) The mean "u"
                is not in support space "x".
                Choose "u" in "x"."""
            )
        else:
            # when k -> +inf von mises
            # are delta functions
            if v_k[0] > 713:

                # make the first a delta
                first_vm = np.zeros((len(v_x)))
                first_vm[x_rad == u_rad[0]] = 1

                # get others by circular
                # translation of the first
                vmises = self._shift_circular(
                    v_x, v_u[1:], first_vm, v_u[0],
                )
            else:
                # calculate the first
                first_vm = self._calculate_von_mises(
                    x_rad, u_rad[0], v_k[0]
                )

                # get others by circular
                # translation of the first
                vmises = self._shift_circular(
                    v_x, v_u[1:], first_vm, v_u[0],
                )

                # normalize to probabilities
                if self.p:
                    vmises = vmises / sum(vmises[:, 0])
            return vmises

    def _shift_circular(
        self, v_x, v_u, first_vmises, first_mean
    ):
        """translate von mises circularly
        by v_u - first mean
        to create a matrix of von mises

        Args:
            v_x (_type_): _description_
            v_u (_type_): _description_
            first_vmises (_type_): _description_
            first_mean (_type_): _description_

        Returns:
            _type_: _description_
        """

        # initialize von mises matrix
        vmises = np.zeros((len(v_x), len(v_u) + 1))
        vmises[:, 0] = first_vmises
        rot_start = np.where(v_x == first_mean)[0]
        for i in range(len(v_u)):
            rot_end = np.where(v_x == v_u[i])[0]
            rotation = rot_end - rot_start
            vmises[:, i + 1] = np.roll(
                vmises[:, 0], rotation
            )
        return vmises

    def _get_different_k_and_means(
        self, x_rad: np.array, u_rad: np.array, v_k: list
    ):
        """get von mises for different concentrations k and 
        means u

        Args:
            x_rad (np.array): von mises() support space
            u_rad (np.array): von mises means in radians
            v_k (list): von mises concentrations

        Returns:
            np.array: von mises array of
                (Nx x_rad) rows, (Nm mean * Nk) cols  
        """
        vmises_all = []
        for u_i in u_rad:
            for k_i in v_k:
                vmises = self._calculate_von_mises(
                    x_rad, u_i, k_i
                )
                # normalize
                if self.p:
                    vmises = vmises / sum(vmises)

                # store
                vmises_all.append(vmises)
        return np.array(vmises_all).T

    def _calculate_von_mises(
        self, x_rad: np.array, u_rad: float, v_k: float
    ):
        """Calculate a von Mises function

        Args:
            x_rad (np.array): 
                von mises' support space in radians
            u_rad (float): 
                von mises means in radians
            v_k (float): von mises concentrations

        Returns:
            np.array: f(x_rad,u_rad,k_rad)
        """
        # handle exceptions
        if v_k > 713:
            raise ValueError(
                """v_k must be < 713
                else, bessel function=inf
                """
            )

        # calculate von mises
        amp = 1
        bessel_order = 0
        scaling = 2 * pi * iv(bessel_order, v_k)
        vm_fun = exp(v_k * cos(amp * (x_rad - u_rad)) - v_k)
        vmises = vm_fun / scaling
        return vmises

    def _get_deg_to_rad(self, deg: np.array, signed: bool):
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


class VonMisesMixture:
    """Mixture of Von Mises class"""

    def __init__(self, p: bool):
        """instantiate VonMisesMixture class

        Args:
            p (bool): True/False means probabilities 
            or not
        """
        self.p = p

    def get(
        self,
        support_space: np.ndarray,
        vm_mean: np.ndarray,
        vm_k: list,
        mixt_coeff: float,
    ) -> np.ndarray:
        """Calculate mixture of von mises

        Args:
            support_space (np.ndarray): _description_
            vm_mean (np.ndarray): _description_
            vm_k (list): _description_
            mixt_coeff (float): _description_

        Returns:
            np.ndarray: mixture of von Mises

        Usage:

            .. code-block:: python
            
            vm_mixt = VonMisesMixture(p=True)
            support_space = np.arange(0,360,1)
            vm_means = np.array([0,90,180,270])
            vm_k = [2.7, 2.7, 2.7, 2.7]
            mixture_coeff = 0.25
            mixture = vm_mixt.get(support_space, vm_means, vm_k, mixture_coeff)

        """
        # calculate all von mises
        vm_proba = VonMises(p=True).get(
            support_space, vm_mean, vm_k
        )
        # combine them
        mixture = mixt_coeff * np.sum(vm_proba, 1)
        return mixture


def is_unique(x):
    """Check if x contains
    repetition of one value

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    return len(np.unique(x)) == 1
