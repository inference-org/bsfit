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
import sys

import yaml
from matplotlib import pyplot as plt

from bsfit.nodes.dataEng import (
    simulate_dataset,
    simulate_small_dataset,
    simulate_task_conditions,
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
GRANULARITY = "trial"
CENTERING = True
CASE = 1

# simulate case 0
if CASE == 0:
    INIT_P = {
        "k_llh": [33],
        "k_prior": [0, 33],
        "k_card": [0],
        "prior_tail": [0],
        "p_rand": [0],
        "k_m": [2000],
    }
elif CASE == 1:
    # simulate case 1
    PRIOR_NOISE = [80, 40]
    STIM_NOISE = [0.33, 0.66, 1.0]
    INIT_P = {
        "k_llh": [2.7, 10.7, 33],
        "k_prior": [2.7, 33],
        "k_card": [0],
        "prior_tail": [0],
        "p_rand": [0],
        "k_m": [2000],
    }

if __name__ == "__main__":
    """Entry point that runs analytical pipelines
    e.g., for now fiiting the standard bayesian model 
    and generating predictions
    
    Cli args:
    - "fit_standard_bayes"
    - "simulate_standard_bayes_trial_data"
    
    Usage:
    .. code-block:: console
    
        python main.py fit_standard_bayes
    """

    if sys.argv[1] == "fit_standard_bayes":

        # simulate a dataset
        logger.info("Simulating dataset ...")

        # simulate case 0
        if CASE == 0:
            dataset = simulate_small_dataset()
        elif CASE == 1:
            # simulate case 1
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

        # get the test dataset
        test_dataset = get_data(dataset)

        # calculate predictions
        output = model.predict(
            test_dataset, granularity=GRANULARITY
        )

        # calculate data and prediction statistics
        estimate = test_dataset[1]
        output = get_data_stats(estimate, output)

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

    elif sys.argv[1] == "simulate_data_by_standard_bayes":

        # set pipeline parameters
        GRANULARITY = "trial"
        SIM_P = {
            "k_llh": [2.7, 10.7, 33],
            "k_prior": [2.7, 33],
            "k_card": [0],
            "prior_tail": [0],
            "p_rand": [0],
            "k_m": [2000],
        }
        # simulate a dataset
        logger.info("simulating dataset ...")

        # simulate task conditions
        # (dataset design matrix)
        conditions = simulate_task_conditions(
            stim_noise=STIM_NOISE,
            prior_mode=PRIOR_MODE,
            prior_noise=PRIOR_NOISE,
            prior_shape=PRIOR_SHAPE,
        )

        # instantiate model
        model = StandardBayes(
            prior_shape=PRIOR_SHAPE,
            prior_mode=PRIOR_MODE,
            readout=READOUT,
        )

        # simulate trial predictions
        # stochastically
        output = model.simulate(
            dataset=conditions,
            sim_p=SIM_P,
            granularity=GRANULARITY,
            centering=CENTERING,
        )

        # calculate prediction statistics
        plt.figure(figsize=(15, 5))
        stat_out = model.simulate(
            dataset=output["dataset"],
            sim_p=SIM_P,
            granularity="mean",
            centering=CENTERING,
        )
        # print dataset
        logger.info("Printing simulated trial dataset ...")
        logger.info(output["dataset"])
