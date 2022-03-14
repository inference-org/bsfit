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
    stim_std: float, prior_mode: float, prior_std: float, prior_shape:str
):
    """Simulate a test database

    Returns:
        (pd.DataFrame): _description_
    """

    # initialize dataframe
    data = pd.DataFrame()

    # set stimulus mean
    data["stim_mean"] = np.arange(0, 360, 1)

    # set stimulus std
    data["stim_std"] = np.repeat(
        stim_std, len(data["stim_mean"])
    )

    # set prior mode
    data["prior_mode"] = np.repeat(
        prior_mode, len(data["stim_mean"])
    )

    # set prior std
    data["prior_std"] = np.repeat(
        prior_std, len(data["stim_mean"])
    )

    # set prior std
    data["prior_shape"] = np.repeat(
        prior_shape, len(data["stim_mean"])
    )

    # simulate estimate choices
    data["estimate"] = np.arange(0, 360, 1)

    print(data)
    return data
