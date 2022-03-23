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

from ..utils import fit_maxlogl


def fit(
    database: pd.DataFrame,
    prior_shape: str,
    prior_mode: float,
    readout: str,
):
    """fit model

    Args:
        database (pd.DataFrame): _description_
        prior_shape (str): prior shape
        - "vonMisesPrior"
        prior_mode (float): prior mode
        readout (str): the posterior readout
        - "map": maximum a posteriori

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

