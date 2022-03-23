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

import logging
import logging.config
import os

import yaml

from bsfit.nodes.dataEng import make_dataset, simulate_dataset
from bsfit.nodes.models.bayes import StandardBayes
from bsfit.nodes.utils import get_data

# setup logging
proj_path = os.getcwd()
logging_path = os.path.join(proj_path + "/conf/logging.yml")

with open(logging_path, "r") as f:
    LOG_CONF = yaml.load(f, Loader=yaml.FullLoader)

logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger(__name__)

# set parameters
SUBJECT = "sub01"
PRIOR_SHAPE = "vonMisesPrior"
PRIOR_MODE = 225
OBJ_FUN = "maxLLH"
READOUT = "map"
PRIOR_NOISE = [80, 40]  # e.g., prior's std
STIM_NOISE = [0.33, 0.66]  # e.g., motion's coherence


if __name__ == "__main__":
    """Entry point that runs analyses pipelines
    e.g., for now the standard bayesian model fitting
    and predictions
    """
    # simulate a dataset
    logger.info("Simulating dataset ...")
    dataset = simulate_dataset(
        stim_noise=STIM_NOISE,
        prior_mode=PRIOR_MODE,
        prior_noise=PRIOR_NOISE,
        prior_shape=PRIOR_SHAPE,
    )

    # train model
    logger.info("Fitting bayes model ...")

    # instantiate model
    model = StandardBayes(
        prior_shape=PRIOR_SHAPE,
        prior_mode=PRIOR_MODE,
        readout=READOUT,
    )

    # train model
    model = model.fit(dataset=dataset)

    # print results
    logger.info("Printing fitting results ...")
    logger.info(
        f"""best fit params: {model.best_fit_p} 
        - neglogl: {model.neglogl}"""
    )

    # get test dataset
    test_dataset = get_data(dataset)

    # get predictions
    output = model.predict(
        test_dataset, granularity="trial"
    )
    logger.info("Printing predict results ...")
    logger.info(output.keys())

    # done
    logger.info("Done.")
