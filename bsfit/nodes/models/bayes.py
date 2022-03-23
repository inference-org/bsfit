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

from ..utils import fit_maxlogl, predict


class StandardBayes:
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

    def fit(self, dataset: pd.DataFrame):
        """fit the model

        Args:
            dataset (pd.DataFrame): _description_

        Returns:
            (StandarBayes): fitted model
        """
        print("Training the model ...\n")
        output = fit_maxlogl(
            dataset,
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

