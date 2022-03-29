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


import pandas as pd
from bsfit.nodes.utils import get_data, get_data_stats
from bsfit.nodes.viz.prediction import plot_mean

from ..utils import fit_maxlogl, predict, simulate, simulate_dataset


class StandardBayes:
    """Standard Bayesian model
    """

    def __init__(
        self,
        prior_shape: str,
        prior_mode: float,
        readout: str,
    ):
        """Instantiate Standard Bayesian model

        Args:
            prior_shape (str): prior shape
            - "vonMisesPrior"
            prior_mode (float): prior mode
            readout (str): the posterior readout
            - "map": maximum a posteriori
        """
        self.prior_shape = prior_shape
        self.prior_mode = prior_mode
        self.readout = readout
        self.best_fit_p = []
        self.neglogl = []
        self.params = []

    def fit(self, dataset: pd.DataFrame, init_p: dict):
        """fit the model

        Args:
            dataset (pd.DataFrame): _description_
            init_p (dict): _description_

        Returns:
            (StandarBayes): fitted model
        """
        print("Training the model ...\n")
        output = fit_maxlogl(
            dataset,
            init_p,
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
        sim_p: dict,
        granularity: str,
        centering: bool,
        **kwargs: dict,
    ):
        """simulate predictions

        Args:
            dataset (pd.DataFrame): dataset
            - either task conditions by columns
            - or task conditions and data ("estimate")
            sim_p (dict): _description_
            granularity (str): _description_
            centering (bool): _description_
        
        Kwargs:
            when granularity="trial":
            - n_repeats (int): the number of repeats of 
            each task condition
                
        Returns:
            (dict): simulation results
        """
        print("Running simulation ...\n")

        # calculate data must be overlapped with
        # model simulations
        output = simulate(
            dataset,
            sim_p,
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
                **kwargs
            )
        return output

    def predict(
        self, data: tuple, granularity: str
    ) -> dict:
        """Calculate the model's predictions

        Args:
            data (tuple): the data to predict
            - tuple of (stim_mean, stim_estimate) 
            granularity (str): predict single trial or mean estimate
            - "trial"

        Returns:
            (dict): _description_
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

