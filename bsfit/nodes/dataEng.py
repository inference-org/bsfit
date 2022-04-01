# note: the doctsring code below within
# """ is converted to a restructuredText
# .rst file by sphinx to automatically
# generate the api's documentation
#
# docstring style used: Google style
"""
    Data engineering module

    Copyright 2022 by Steeve Laquitaine, GNU license 
"""

import os
from time import time

import numpy as np
import pandas as pd
import scipy.io


def load_mat(file_path: str):
    """load matlab file

    Args:
        file_path (str): file path

    Returns:
        _type_: _description_
    
    Usage:
        .. code-block:: python
            
            file_path = "data/data01_direction4priors/data/sub01/steeve_exp11_data_sub01_sess01_run01_Pstd010_mean225_coh006_012_024_dir36_t107_73_33perCoh_130217_lab.mat"
            file = scipy.io.loadmat(file_path)
    """
    return scipy.io.loadmat(file_path)


def make_dataset(subject: str, data_path: str, prior: str):
    """load and engineer dataset [TODO]

    Args:
        subject (str): [TODO]
        data_path (str): [TODO]
        prior (str): [TODO]

    Usage:
        .. code-block:: python        
        
            dataset = make_dataset(
                subject='sub01',
                data_path='data/',...
                prior='vonMisesPrior'
                )    

    Returns:
        dataset:
    """
    # time
    t0 = time()

    # go to data path
    os.chdir(data_path)

    # loop over subjects
    return None


def simulate_small_dataset():
    """simulate a case of prior-induced bias in circular estimate with one 
    stimulus noise and two prior noises

    Returns:
        (pd.DataFrame): a design matrix of task conditions
    """
    # init df
    data = pd.DataFrame()

    # set stimulus mean (e.g., 5 to 355)
    data["stim_mean"] = np.array(
        [
            200,
            210,
            220,
            230,
            240,
            250,
            200,
            210,
            220,
            230,
            240,
            250,
        ]
    )

    # set stimulus std
    data["stim_std"] = np.array(
        [
            0.66,
            0.66,
            0.66,
            0.66,
            0.66,
            0.66,
            0.66,
            0.66,
            0.66,
            0.66,
            0.66,
            0.66,
        ]
    )

    # set prior mode
    data["prior_mode"] = np.array(
        [
            225,
            225,
            225,
            225,
            225,
            225,
            225,
            225,
            225,
            225,
            225,
            225,
        ]
    )

    # set prior std
    data["prior_std"] = np.array(
        [80, 80, 80, 80, 80, 80, 20, 20, 20, 20, 20, 20,]
    )

    # set prior std
    data["prior_shape"] = np.array(
        [
            "vonMisesPrior",
            "vonMisesPrior",
            "vonMisesPrior",
            "vonMisesPrior",
            "vonMisesPrior",
            "vonMisesPrior",
            "vonMisesPrior",
            "vonMisesPrior",
            "vonMisesPrior",
            "vonMisesPrior",
            "vonMisesPrior",
            "vonMisesPrior",
        ]
    )

    # simulate estimate choices
    data["estimate"] = np.array(
        [
            200,
            210,
            220,
            230,
            240,
            250,
            218,
            220,
            223,
            227,
            230,
            233,
        ]
    )
    return data


def simulate_dataset(
    stim_noise: float,
    prior_mode: float,
    prior_noise: float,
    prior_shape: str,
):
    """simulate a test dataset (a design matrix)
    
    Args:
        stim_noise (float): stimulus noise conditions (e.g., motion coherence)
        prior_mode (float): prior mode conditions (e.g., 225)
        prior_noise (float): prior noise conditions (e.g. std)
        prior_shape (str): prior function condition (e.g., "vonMisesPrior")

    Returns:
        (pd.DataFrame): a design matrix of task conditions
    """

    # initialize dataframe
    data = pd.DataFrame()

    # loop over stimulus and prior
    # noises to simulate task conditions
    for stim_noise_i in stim_noise:
        for prior_noise_i in prior_noise:

            # init df
            df = pd.DataFrame()

            # set stimulus mean (e.g., 5 to 355)
            df["stim_mean"] = np.arange(5, 365, 5)

            # set stimulus std
            df["stim_std"] = np.repeat(
                stim_noise_i, len(df["stim_mean"])
            )

            # set prior mode
            df["prior_mode"] = np.repeat(
                prior_mode, len(df["stim_mean"])
            )

            # set prior std
            df["prior_std"] = np.repeat(
                prior_noise_i, len(df["stim_mean"])
            )

            # set prior std
            df["prior_shape"] = np.repeat(
                prior_shape, len(df["stim_mean"])
            )

            # simulate estimate choices (0 to 359)
            df["estimate"] = df["stim_mean"]

            # record
            data = pd.concat([data, df])

    return data


def simulate_task_conditions(
    stim_noise: float,
    prior_mode: float,
    prior_noise: float,
    prior_shape: str,
):
    """simulate task conditions (a design matrix)

    Args:
        stim_noise (float): stimulus noise conditions (e.g., motion coherence)
        prior_mode (float): prior mode conditions (e.g., 225)
        prior_noise (float): prior noise conditions (e.g. std)
        prior_shape (str): prior function condition (e.g., "vonMisesPrior")

    Returns:
        pd.DataFrame: a design matrix of task conditions
    """

    # initialize dataframe
    conditions = pd.DataFrame()

    # loop over stimulus and prior
    # noises to simulate task conditions
    for stim_noise_i in stim_noise:
        for prior_noise_i in prior_noise:

            # init df
            df = pd.DataFrame()

            # set stimulus mean (e.g., 5 to 355)
            df["stim_mean"] = np.arange(5, 365, 5)

            # set stimulus std
            df["stim_std"] = np.repeat(
                stim_noise_i, len(df["stim_mean"])
            )

            # set prior mode
            df["prior_mode"] = np.repeat(
                prior_mode, len(df["stim_mean"])
            )

            # set prior std
            df["prior_std"] = np.repeat(
                prior_noise_i, len(df["stim_mean"])
            )

            # set prior std
            df["prior_shape"] = np.repeat(
                prior_shape, len(df["stim_mean"])
            )

            # record
            conditions = pd.concat([conditions, df])
    return conditions
