# note: the doctsring code below within
# """ is converted to a restructuredText
# .rst file by sphinx to automatically
# generate the api's documentation
#
# docstring style used: Google style
"""""
This is the software's entry point.

    usage:

    .. highlight:: python
    .. code-block:: python
    
        python main.py

Copyright 2022 by Steeve Laquitaine, GNU license 
"""

from src.nodes.dataEng import (
    make_database,
    simulate_database,
)
from src.nodes.models import bayes

# setup parameters
DATA_PATH = "data/data01_direction4priors/data/"
SUBJECT = "sub01"
PRIOR_SHAPE = "vonMisesPrior"
PRIOR_MODE = 225
OBJ_FUN = "maxLLH"
READOUT = "map"
PRIOR_STD = 80
STIM_STD = 0.33

if __name__ == "__main__":
    """Entry point
    """
    # simulate a database
    database = simulate_database(
        stim_std=STIM_STD,
        prior_mode=PRIOR_MODE,
        prior_std=PRIOR_STD,
        prior_shape=PRIOR_SHAPE,
    )

    # train model
    output = bayes.fit(
        database=database,
        data_path=DATA_PATH,
        prior_shape=PRIOR_SHAPE,
        prior_mode=PRIOR_MODE,
        readout=READOUT,
    )

    # print results
    print(output)
