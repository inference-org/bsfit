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

import os
from time import time

import numpy as np
import pandas as pd
import scipy.io


def load_mat(file_path: str):
    """load matlab file

    Args:
        file_path (str): _description_

    Returns:
        _type_: _description_
    
    Usage:
        file = scipy.io.loadmat("data/data01_direction4priors/data/sub01/steeve_exp11_data_sub01_sess01_run01_Pstd010_mean225_coh006_012_024_dir36_t107_73_33perCoh_130217_lab.mat")
    """
    return scipy.io.loadmat(file_path)


def make_database(subject: str, data_path: str, prior: str):
    """Load and engineer database

    Args:
        subject (str): _description_
        data_path (str): _description_
        prior (str): _description_
    
    Returns:
        database():

    Usage:
               
        database = make_database(
            subject='sub01',
            data_path='data/',...
            prior='vonMisesPrior'
            )
    """
    # time
    t0 = time()

    # go to data path
    os.chdir(data_path)

    # loop over subjects

    pass


def simulate_database(
    stim_noise: float,
    prior_mode: float,
    prior_noise: float,
    prior_shape: str,
):
    """Simulate a test database

    Returns:
        (pd.DataFrame): _description_
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
