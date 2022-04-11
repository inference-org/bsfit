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
from matplotlib import pyplot as plt

from bsfit.nodes.config import parametrize_pipe
from bsfit.nodes.dataEng import (
    simulate_dataset,
    simulate_small_dataset,
    simulate_task_conditions,
)
from bsfit.nodes.models.bayes import (
    CardinalBayes,
    StandardBayes,
)
from bsfit.nodes.models.utils import (
    get_data,
    get_data_stats,
)
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
PRIOR_SHAPE = "vonMisesPrior"
PRIOR_MODE = 225
OBJ_FUN = "maxLLH"
READOUT = "map"
GRANULARITY = "trial"
CENTERING = True
CASE = 1


if __name__ == "__main__":
    """It is the entry point.
    
    Usage:

    .. code-block:: console
        
        # to see arguments run 
        python main.py -h

        # e.g., 
        python main.py --model standard_bayes --analysis fit
    """

    # select and parametrize a pipeline
    args = parametrize_pipe()

    if (
        args.model == "standard_bayes"
        and args.analysis == "fit"
    ):

        # simulate case 0
        if CASE == 0:
            INIT_P = {
                "k_llh": [33],
                "k_prior": [0, 33],
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
                "p_rand": [0],
                "k_m": [2000],
            }

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
            initial_params=INIT_P,
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

    elif (
        args.model == "cardinal_bayes"
        and args.analysis == "fit"
    ):

        # case 0
        if CASE == 0:
            INIT_P = {
                "k_llh": [33],
                "k_prior": [0, 33],
                "k_card": [2000],
                "p_rand": [0],
                "k_m": [2000],
            }
            # case 1
        elif CASE == 1:
            PRIOR_NOISE = [80, 40]
            STIM_NOISE = [0.33, 0.66, 1.0]
            INIT_P = {
                "k_llh": [2.7, 10.7, 33],
                "k_prior": [2.7, 33],
                "k_card": [2000],
                "p_rand": [0],
                "k_m": [2000],
            }

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
        logger.info("Fitting cardinal bayesian model ...")

        # instantiate model
        model = CardinalBayes(
            initial_params=INIT_P,
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

    elif (
        args.model == "standard_bayes"
        and args.analysis == "simulate_data"
    ):

        # set pipeline parameters
        # - SIM_P: simulation parameters
        # - N_REPEATS: number of repetition of
        # task each condition
        PRIOR_NOISE = [80, 40]
        STIM_NOISE = [0.33, 0.66, 1.0]
        SIM_P = {
            "k_llh": [2.7, 5.7, 11],
            "k_prior": [2.7, 11],
            "prior_tail": [0],
            "p_rand": [0],
            "k_m": [2000],
        }
        GRANULARITY = "trial"
        N_REPEATS = 5

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
            initial_params=SIM_P,
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
            n_repeats=N_REPEATS,
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

    elif (
        args.model == "cardinal_bayes"
        and args.analysis == "simulate_data"
    ):

        # set pipeline parameters
        # - SIM_P: simulation parameters
        # - N_REPEATS: number of repetition of
        # task each condition
        PRIOR_NOISE = [80, 40]
        STIM_NOISE = [0.33, 0.66, 1.0]
        SIM_P = {
            "k_llh": [2.7, 5.7, 11],
            "k_prior": [2.7, 11],
            "k_card": [2000],
            "prior_tail": [0],
            "p_rand": [0],
            "k_m": [2000],
        }
        GRANULARITY = "trial"
        N_REPEATS = 5

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
        model = CardinalBayes(
            initial_params=SIM_P,
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
            n_repeats=N_REPEATS,
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
