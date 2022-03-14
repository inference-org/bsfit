"""
SLfitBayesianModel.m


     author: steeve laquitaine
       date: 131205 updated 160113
     status: Complete
    purpose: Model motion direction estimation data as Bayesian
             Also fit other Bayesian models

 References:
     -Hurliman et al, 2002,VR
     -Stocker&Simoncelli,2006,NN
     -Girshick&Simoncelli,2011,NN
     -Chalk&Series,2012,JoV
"""

from time import time

from src.nodes.dataEng import make_database, simulate_database
from src.nodes.utils import fit_maxlogl


def fit(
    subject: str,
    data_path: str,
    prior_shape: str,
    readout: str,
    objfun: str,
):

    # time
    t0 = time()

    # create database [TODO]
    # database = make_database(subject, data_path, prior)
    database = simulate_database(
        stim_std=0.33,
        prior_mode=225,
        prior_std=80,
        prior_shape=prior_shape,
    )

    # fit
    neg_logl = fit_maxlogl(database)

    output = t0
    return output

