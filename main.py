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

from bsfit.nodes.dataEng import (
    make_dataset,
    simulate_dataset,
)
from bsfit.nodes.models.bayes import StandardBayes
from bsfit.nodes.utils import get_data, get_data_stats
from bsfit.nodes.viz.prediction import plot_mean

# setup logging
proj_path = os.getcwd()
logging_path = os.path.join(proj_path + "/conf/logging.yml")

with open(logging_path, "r") as f:
    LOG_CONF = yaml.load(f, Loader=yaml.FullLoader)

logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger(__name__)

# set parameters
# - PRIOR_NOISE: e.g., an object's motion direction density's std
# - STIM_NOISE: e.g., an object's motion coherence
# - CENTERING: center or not plot relative to prior mode
SUBJECT = "sub01"
PRIOR_SHAPE = "vonMisesPrior"
PRIOR_MODE = 225
OBJ_FUN = "maxLLH"
READOUT = "map"
PRIOR_NOISE = [80, 40]
STIM_NOISE = [0.33, 0.66, 1.0]
INIT_P = {
    "k_llh": [1, 1, 1],
    "k_prior": [1, 1],
    "k_card": [1],
    "prior_tail": [0],
    "p_rand": [0],
    "k_m": [0],
}
CENTERING = True

if __name__ == "__main__":
    """Entry point that runs analytical pipelines
    e.g., for now fiiting the standard bayesian model 
    and generating predictions
    """
    # simulate a dataset
    logger.info("Simulating dataset ...")
    dataset = simulate_dataset(
        stim_noise=STIM_NOISE,
        prior_mode=PRIOR_MODE,
        prior_noise=PRIOR_NOISE,
        prior_shape=PRIOR_SHAPE,
    )

    # log status
    logger.info("Fitting bayes model ...")

    # instantiate model
    model = StandardBayes(
        prior_shape=PRIOR_SHAPE,
        prior_mode=PRIOR_MODE,
        readout=READOUT,
    )

    # train model
    model = model.fit(dataset=dataset, init_p=INIT_P)

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

    # get data and prediction stats
    estimate = test_dataset[1]
    output2 = get_data_stats(estimate, output)

    # plot data and prediction mean
    plot_mean(
        output["data_mean"],
        output["data_std"],
        output["prediction_mean"],
        output["prediction_std"],
        output["conditions"],
        prior_mode=PRIOR_MODE,
        centering=CENTERING,
    )

    # log status
    logger.info("Printing predict results ...")
    logger.info(output.keys())

    # done
    logger.info("Done.")
