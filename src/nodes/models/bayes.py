#      author: steeve laquitaine
#     purpose: Model motion direction estimation data as Bayesian
#              Also fit other Bayesian models
#  References:
#      -Hurliman et al, 2002,VR
#      -Stocker&Simoncelli,2006,NN
#      -Girshick&Simoncelli,2011,NN
#      -Chalk&Series,2012,JoV


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
        database, prior_shape, prior_mode, readout
    )
    return output

