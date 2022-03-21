# note: the doctsring code below within
# """ is converted to a restructuredText
# .rst file by sphinx to automatically
# generate the api's documentation
#
# docstring style used: Google style
"""
    fit estimation data with a Bayesian model

    see:
        Hurliman et al, 2002,VR
        Stocker&Simoncelli,2006,NN
        Girshick&Simoncelli,2011,NN
        Chalk&Series,2012,JoV

    Copyright 2022 by Steeve Laquitaine, GNU license 
"""


from time import time

import pandas as pd
from src.nodes.utils import fit_maxlogl


def fit(
    database: pd.DataFrame,
    data_path: str,
    prior_shape: str,
    prior_mode: float,
    readout: str,
):
    """fit model

    Args:
        database (pd.DataFrame): _description_
        data_path (str): _description_
        prior_shape (str): prior shape
        prior_shape (float): prior mode
        readout (str): _description_

    Returns:
        _type_: _description_
    """

    # time
    t0 = time()

    # fit
    output = fit_maxlogl(
        database, prior_shape, prior_mode, readout,
    )
    return output

