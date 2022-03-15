from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import fmin
from src.nodes.data import VonMises


def fit_maxlogl(data):
    """fit maximum log(likelihood)

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    # make 360 and 0 degrees same
    data["estimate"][data["estimate"] == 0] = 360

    # estimate = data["estimate"]

    # fitPbkp = slfitBayesfminSlogl(
    #     k0,
    #     data["estimate"],
    #     data["stim_mean"],
    #     data["stim_std"],
    #     data["prior_std"],
    #     data["prior_shape"],
    #     data["prior_mode"],
    #     TheModel,
    #     options,
    #     varargin{:}
    #     )

    # fitPtmp,negLogl,exitflag,outputFit = fminsearch(@(fitPtmp) ...
    #     SLgetLoglBayesianModel(data["estimate"],...
    #     disp,...
    #     StimStrength,...
    #     pstd,...
    #     fitPtmp,...
    #     priorShape,...
    #     priorModes,...
    #     TheModel,varargin{:}),...
    #     initp(i,:),...
    #     options);

    # TEST ==========
    out = test()

    # REAL ========
    # select initial model parameters
    K_LLH = [1]
    K_PRIOR = [1]
    K_CARD = [1]
    PRIOR_TAIL = [0]

    # set parameters
    model_params = [K_LLH, K_PRIOR, K_CARD, PRIOR_TAIL]
    task_params = (
        (
            data["stim_std"],
            data["prior_mode"],
            data["prior_std"],
            data["prior_shape"],
        ),
    )

    # set data
    true = data["stim_mean"]
    pred = data["estimate"]

    # fit
    neglogl = fmin(
        get_logl,
        model_params,
        args=(true, pred, task_params),
    )
    print(neglogl[0])
    return neglogl


def test():
    """_summary_

    Returns:
        _type_: _description_
    """
    # parametric function, x is the independent variable
    # and c are the parameters.
    # it's a polynomial of degree 2
    fp = lambda c, x: c[0] + c[1] * x + c[2] * x * x
    real_p = np.random.rand(3)

    # error function to minimize
    e = lambda p, x, y: (abs((fp(p, x) - y))).sum()

    # generating data with noise
    n = 30
    x = np.linspace(0, 1, n)
    y = fp(real_p, x) + np.random.normal(0, 0.05, n)

    # fitting the data with fmin
    p0 = np.random.rand(3)  # initial parameter value
    p = fmin(e, p0, args=(x, y))
    return p


def get_logl(
    model_params: list,
    true: pd.Series,
    pred: pd.Series,
    task_params: tuple,
):
    """_summary_

    Args:
        model_params (list): _description_
        true (pd.Series): _description_
        pred (pd.Series): _description_
        task_params (tuple): _description_

    Returns:
        _type_: _description_
    """

    # get task params
    (
        stim_std,
        prior_mode,
        prior_std_set,
        prior_shape,
    ) = task_params[0]
    stim_mean_set = np.unique(true)
    stim_std_set = np.unique(stim_std)
    n_stim_std = len(stim_std_set)
    n_prior_std = len(np.unique(prior_std_set))

    # get model params
    model_params = model_params.tolist()
    k_llh = model_params[:n_stim_std]
    del model_params[:n_stim_std]
    k_prior = model_params[:n_prior_std]
    del model_params[:n_prior_std]
    k_cardinal = model_params.pop(0)
    prior_tail = model_params

    # set percept space
    percept_space = np.arange(0, 360, 1)

    # init outputs
    llh_map = defaultdict(dict)

    # store by prior std
    for i in range(len(prior_std_set)):
        for j in range(n_stim_std):

            # record stimulus strength
            # vrg{:}{numVarg+2} = stim_noise[j]

            # compute percept distribution
            # map: maximum a posteriori readouts
            map, llh_map[i][j] = get_bayes_lookup(
                percept_space,
                stim_mean_set,
                k_llh[j],
                prior_mode,
                k_prior[i],
                k_cardinal,
                prior_tail,
                prior_shape,
            )
    return None


def get_bayes_lookup(
    percept_space: np.array,
    stim_mean: np.array,
    k_llh: float,
    prior_mode: float,
    k_prior: float,
    k_cardinal: float,
    prior_tail: float,
    prior_shape: str,
):
    """Create a bayes lookup matrix
    based on Girshick's paper
    M measurements in rows
    x N stimulus feature means in columns

    usage:

        percept, logl_percept = get_bayes_lookup(
            percept_space=1:1:360,
            stim_mean=5:10:355,
            k_llh=5,
            prior_mode=225,
            k_prior=4.77,
            k_cardinal=0,
            prior_tail=0,
            prior_shape='von_mises',
            )
    """
    # calculate measurement densities ~v(di,kl)
    # di are stimulus mean in columns
    # over range of possible measurement in rows
    meas_density = VonMises(p=True).get(
        percept_space, stim_mean, [k_llh]
    )
    return None


def is_unique(x):
    return len(np.unique(x)) == 1
