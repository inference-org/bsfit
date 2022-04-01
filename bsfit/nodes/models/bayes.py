# note: the doctsring code below within
# """ is converted to a restructuredText
# .rst file by sphinx to automatically
# generate the api's documentation
#
# docstring style used: Google style
"""
    Bayesian estimators to model circular estimation data

    see:
        Hurliman et al, 2002,VR
        Stocker&Simoncelli,2006,NN
        Girshick&Simoncelli,2011,NN
        Chalk&Series,2012,JoV

    Copyright 2022 by Steeve Laquitaine, GNU license 
"""


from typing import Dict

import pandas as pd
from bsfit.nodes.viz.prediction import plot_mean

from .abstract_model import Model
from .utils import (fit_maxlogl, get_data, get_data_stats, predict, simulate,
                    simulate_dataset)


class StandardBayes(Model):
    """Standard Bayesian model class
    """

    def __init__(
        self,
        initial_params: Dict[str, list],
        prior_shape: str,
        prior_mode: float,
        readout: str,
    ):
        """instantiate Standard Bayesian model

        Args:
            initial_params (Dict): ::

                initial_params = {
                    "k_llh": list,
                    "k_prior": list,
                    "p_rand": list,
                    "k_m": list,
                }

            prior_shape (str): prior's function ("vonMisesPrior")
            prior_mode (float): prior's mode (e.g., 225)
            readout (str): decision process ("map": max a posteriori readout)
        """
        # inherit from parent
        super().__init__()

        # parametrize
        self.initial_params = initial_params
        self.prior_shape = prior_shape
        self.prior_mode = prior_mode
        self.readout = readout
        self.neglogl = None
        self.params = None

    def fit(self, dataset: pd.DataFrame):
        """fit the model

        Args:
            dataset (pd.DataFrame): dataset with columns::
                
                'stim_mean': the stimulus feature (e.g., motion direction)
                'stim_std': the stimulus feature strength 
                'prior_mode': mode of prior
                'prior_std': std of the prior
                'prior_shape': density function of the prior (e.g., "vonMisesPrior")
                'estimate': the estimate data

            init_p (dict): initial parameters

        Usage:
            .. code-block:: python

                from bsfit.nodes.models.bayes import StandardBayes
                initial_params = {
                    "k_llh": [2.7, 10.7, 33],
                    "k_prior": [2.7, 33],
                    "p_rand": [0],
                    "k_m": [2000],
                }
                model = StandardBayes(
                    initial_params=initial_params,
                    prior_shape="vonMisesPrior",
                    prior_mode=225,
                    readout="map")
                model.fit(dataset)

        Returns:
            (Model): the fitted model
        """
        print("Training the model ...\n")
        output = fit_maxlogl(
            dataset,
            self.initial_params,
            self.prior_shape,
            self.prior_mode,
            self.readout,
        )

        # get fitted parameters
        self.best_fit_p = output["best_fit_p"]
        self.neglogl = output["neglogl"]
        self.params = output["params"]
        print("\nTraining is complete !")
        return self

    def simulate(
        self,
        dataset: pd.DataFrame,
        granularity: str,
        centering: bool,
        **kwargs: dict,
    ):
        """simulate predictions

        Args:
            dataset (pd.DataFrame): dataset::

                'stim_mean': the stimulus feature (e.g., motion direction)
                'stim_std': the stimulus feature strength 
                'prior_mode': the stimulus feature density's mode
                'prior_std': the stimulus feature density's std
                'prior_shape': prior's density function (e.g., "vonMisesPrior")
                'estimate' (optional): the estimate data 

            granularity (str): ("trial" or "mean")
            centering (bool): center or not plot's x-axis
        
        Kwargs:
            n_repeats (int): the number of repeats of conditions, when granularity="trial"
                
        Returns:
            (dict): results of the simulation
        """
        print("Running simulation ...\n")

        # calculate data must be overlapped with
        # model simulations
        output = simulate(
            dataset,
            self.initial_params,
            self.prior_shape,
            self.prior_mode,
            self.readout,
        )

        # record parameters
        self.best_fit_p = output["best_fit_p"]
        self.neglogl = output["neglogl"]
        self.params = output["params"]

        # case data are provided,
        # overlap data and predictions
        if "estimate" in dataset.columns:

            # make predictions
            dataset = get_data(dataset)
            output = self.predict(
                dataset, granularity=granularity
            )

            # case calculate statistics
            if granularity == "mean":
                estimate = dataset[1]
                output = get_data_stats(estimate, output)

                # plot data and prediction mean
                plot_mean(
                    output["data_mean"],
                    output["data_std"],
                    output["prediction_mean"],
                    output["prediction_std"],
                    output["conditions"],
                    prior_mode=self.prior_mode,
                    centering=centering,
                )
                # make stochastic choices
            elif granularity == "trial":
                return output["dataset"]
        else:
            # simulate a dataset
            output = simulate_dataset(
                fit_p=self.best_fit_p,
                params=self.params,
                stim_mean=dataset["stim_mean"],
                granularity=granularity,
                **kwargs,
            )
        return output

    def predict(
        self, data: tuple, granularity: str
    ) -> dict:
        """generate the model's predictions

        Args:
            data (tuple): the data to predict, tuple of (stim_mean, stim_estimate) 
            granularity (str): ("trial" or "mean") to predict single choices or their stats

        Returns:
            (dict): the model predictions
        """
        # get predictions
        print("Calculating predictions ...\n")
        predictions = predict(
            self.best_fit_p,
            self.params,
            *data,
            granularity=granularity,
        )
        return predictions


class CardinalBayes(StandardBayes):
    """Cardinal Bayesian model
    """

    def __init__(
        self,
        initial_params: Dict[str, list],
        prior_shape: str,
        prior_mode: float,
        readout: str,
    ):
        """instantiate the Cardinal Bayesian model

        Args:
            prior_shape (str): prior function ("vonMisesPrior")
            prior_mode (float): the mode of the prior
            readout (str): the decision process ("map")
        
        Raises:
            MissingParameter: "k_card" parameter is missing
        
        Returns:
            (Model): a child of the "StandardBayes" with its attributes
        """
        # inherit from parent
        super().__init__(
            initial_params, prior_shape, prior_mode, readout
        )

        # parametrize
        if not "k_card" in self.initial_params:
            raise TypeError(
                """"k_card", the cardinal prior strength is missing. 
                Please add to the parameters. """
            )

