# note: the doctsring code below within
# """ is converted to a restructuredText
# .rst file by sphinx to automatically
# generate the api's documentation
#
# docstring style used: Google style
"""
    module

    Copyright 2022 by Steeve Laquitaine, GNU license 
"""

from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from numpy import arctan2, cos, sin
from scipy.optimize import fmin

from ..cirpy.data import VonMises, VonMisesMixture
from ..cirpy.utils import (get_circ_conv, get_circ_weighted_mean_std,
                           get_deg_to_rad, get_rad_to_deg)
from ..util import is_empty

pd.options.mode.chained_assignment = None


def fit_maxlogl(
    database: pd.DataFrame,
    init_p: dict,
    prior_shape: str,
    prior_mode: float,
    readout: str,
) -> Dict[str, Any]:
    """Fits observed estimate data of the stimulus
    feature mean with the method of maximum
    log(likelihood). This method searches for the
    model parameters that maximize the log(likeligood)
    of the observed data given the model.

    Args:
        database (pd.DataFrame): database
        init_p (dict): initial parameters
        prior_shape (str): shape of the prior  
        - "vonMisesPrior"  
        prior_mode: (float): mode of the prior  

    Returns:
        Dict[str, Any]: results of the fit
    """

    # set parameters
    # (model and task)
    # [TODO]: rename format_params
    params = format_params(
        database, init_p, prior_shape, prior_mode, readout
    )

    # set the data to fit
    data = get_data(database)

    # fit the model
    output = fmin(
        func=get_logl,
        x0=unpack(params["model"]["init_params"]),
        args=(params, *data),
        disp=True,
        retall=True,  # get solutions after iter
        maxiter=1,  # 100,  # max nb of iterations
        maxfun=1,  # 100,  # max nb of func eval
        # ftol=0.0001,  # objfun convergence
    )

    # get fit results
    best_fit_p = output[0]
    neglogl = output[1]

    return {
        "neglogl": neglogl,
        "best_fit_p": best_fit_p,
        "params": params,
    }


def simulate(
    database: pd.DataFrame,
    sim_p: dict,
    prior_shape: str,
    prior_mode: float,
    readout: str,
) -> Dict[str, Any]:
    """Simulate estimate data per task
    condition

    Args:
        database (pd.DataFrame): database
        sim_p (dict): simulation parameters
        prior_shape (str): shape of the prior  
        - "vonMisesPrior"  
        prior_mode: (float): mode of the prior  

    Returns:
        Dict[str, Any]: record of simulation data
    """

    # set parameters
    # (model and task)
    params = format_params(
        database, sim_p, prior_shape, prior_mode, readout
    )

    # get simulation parameters
    sim_fit_p = np.array(
        unpack(params["model"]["init_params"])
    )
    neglogl = None
    return {
        "neglogl": neglogl,
        "best_fit_p": sim_fit_p,
        "params": params,
    }


def unpack(my_dict: Dict[str, list]) -> list:
    """unpack dict into a flat list

    Args:
        my_dict (Dict[list]): dictionary of list

    Returns:
        (list): flat list
    """
    return flatten([my_dict[keys] for keys in my_dict])


def flatten(x: List[list]) -> list:
    """flattens list of list

    Args:
        x (List[list]): list of list

    Returns:
        list: a list
    """
    return [item for sublist in x for item in sublist]


def format_params(
    database: pd.DataFrame,
    init_p: dict,
    prior_shape: str,
    prior_mode: float,
    readout: str,
) -> dict:
    """Set model and task parameters
    free and fixed

    Args:
        database (pd.DataFrame): _description_
        init_p (dict): _description_
        prior_shape (str): _description_
        prior_mode (float): _description_
        readout (str): _description_

    Returns:
        (dict): _description_
    """

    # set model parameters
    # ....................
    # set initial
    model = dict()
    model["init_params"] = init_p

    # set fixed
    model["fixed_params"] = {
        "prior_shape": prior_shape,
        "prior_mode": prior_mode,
        "readout": readout,
    }

    # set task parameters
    # ....................
    # set fixed
    task = dict()
    task["fixed_params"] = {
        "stim_std": database["stim_std"],
        "prior_std": database["prior_std"],
    }
    return {"model": model, "task": task}


def locate_fit_params(
    params: Dict[str, list]
) -> Dict[str, list]:

    """locate fit parameters
    in parameter dictionary

    Args:
        params (Dict[str, list]): parameters

    Returns:
        dict: dictionary of
        the location of each parameter
        type. e.g., 

        .. code-block::

            {
                'k_llh': [0, 1],
                'k_prior': [2, 3],
                'k_card': [4],
                'prior_tail': [5],
                'p_rand': [6]
            }

    Usage:
        .. code-block::
        
            params = {
                'k_llh': [1, 1],
                'k_prior': [1, 1],
                'k_card': [0],
                'prior_tail': [0],
                'p_rand': [0]
            }
            params_loc = locate_fit_params(params)
    """
    loc = -1
    params_loc = params.copy()

    # loop over parameter types
    for p_type in params:
        params_loc[p_type] = []
        # store each param's index
        for _ in params[p_type]:
            loc += 1
            params_loc[p_type] += [loc]
    return params_loc


def get_data(database: pd.DataFrame):
    """get data to fit

    Args:
        database (pd.DataFrame): _description_

    Returns:
        pd.Series: _description_
    """
    # get stimulus feature mean
    stim_mean = database["stim_mean"]

    # set 0 to 360 deg
    loc_zero = database["estimate"] == 0
    database["estimate"][loc_zero] = 360
    estimate = database["estimate"]
    return stim_mean, estimate


def get_logl(
    fit_p: np.ndarray,
    params: dict,
    stim_mean: pd.Series,
    data: pd.Series,
):
    """calculate the log(likelihood) of the 
    stimulus feature's estimate given the model.
    This function is called by scipy.optimize.fmin().

    Args:
        fit_p (np.ndarray): the model's fit parameters
        params (dict): all the fixed parameters:
        - "task": "fixed_params": the task fixed parameters
        - "model": "fixed_params", "init_params"
        stim_mean (pd.Series): stimulus feature mean
        data (pd.Series): data estimate to fit

    Returns:
        (float): -log(likelihood) of 
            data estimate given model
    """
    # get -logl and intermediate
    # calculated variables
    neglogl, _ = get_fit_variables(
        fit_p, params, stim_mean, data=data
    )
    return neglogl


def get_fit_variables(
    fit_p: np.ndarray,
    params: dict,
    stim_mean: pd.Series,
    **kwargs: dict,
) -> float:
    """calculate the log(likelihood) of the 
    observed stimulus feature mean's estimate
    given the model.

    Args:
        fit_p (np.ndarray): model fit parameters
        params (dict): fixed parameters
        - "task": "fixed_params"
        - "model": "fixed_params", "init_params"
        stim_mean (pd.Series): stimulus feature mean

    Kwargs:
        "data" (pd.Series):: data estimate to fit

    Returns:
        (float): -log(likelihood) of 
        data estimate given model
        (dict): intermediate variables
    """

    # count fit params
    n_fit_params = sum(~np.isnan(fit_p))

    # get fit parameters
    # locate by type
    params_loc = locate_fit_params(
        params["model"]["init_params"]
    )
    k_llh = fit_p[params_loc["k_llh"]]
    k_prior = fit_p[params_loc["k_prior"]]
    p_rand = fit_p[params_loc["p_rand"]][0]
    k_m = fit_p[params_loc["k_m"]][0]

    # set model-dependent parameters
    # set to 0 if not set
    # set cardinal prior strength
    if "k_card" in params_loc:
        k_card = fit_p[params_loc["k_card"]]
    else:
        k_card = 0.0

    # set thickness of prior tail
    if "prior_tail" in params_loc:
        prior_tail = fit_p[params_loc["prior_tail"]][0]
    else:
        prior_tail = 0.0

    # calculate percept probability density
    output = get_proba_percept(
        stim_mean,
        params,
        k_llh,
        k_prior,
        k_card,
        prior_tail,
        p_rand,
    )
    proba_percept = output["PupoGivenModel"]

    # calculate estimate probability density
    # (len(estimate_space) x N_conditions)
    proba_estimate = get_proba_estimate(k_m, proba_percept)

    # when data exist
    if "data" in kwargs:

        # calculate data probability density
        proba_data = get_proba_data(
            kwargs["data"], proba_estimate
        )

        # get the log(likelihood) of the data
        fit_out = get_logl_and_aic(n_fit_params, proba_data)
    else:
        fit_out = dict()
        fit_out["neglogl"] = np.nan
        fit_out["aic"] = np.nan

    # [TODO] setup logging
    print(
        "-logl:{:.2f}, aic:{:.2f}, kl:{}, kp:{}, kc:{}, pt:{:.2f}, pr:{:.2f}, km:{:.2f}".format(
            fit_out["neglogl"],
            fit_out["aic"],
            k_llh,
            k_prior,
            k_card,
            prior_tail,
            p_rand,
            k_m,
        )
    )
    return (
        fit_out["neglogl"],
        {
            "PestimateGivenModel": proba_estimate,
            "map": output["readout_percept"],
            "conditions": output["conditions"],
        },
    )


def get_logl_and_aic(
    n_fit_params: int, proba_data: np.ndarray
) -> dict:
    """calculate log(likelihood) and 
    akaike information criterion

    Args:
        n_fit_params (int): number of fit parameters
        proba_data (np.ndarray): probability of data

    Returns:
        _type_: _description_
    """

    Logl_pertrial = np.log(proba_data)

    # we minimize the objective function
    negLogl = -sum(Logl_pertrial)

    # akaike information criterion metric
    aic = 2 * (n_fit_params - sum(Logl_pertrial))

    return {"neglogl": negLogl, "aic": aic}


def get_proba_data(estimate, proba_estimate):

    # normalize 0 values to 360
    if (estimate == 0).any():
        estimate[estimate == 0] = 360

    # single trial's measurement, its position(row)
    # for each trial(col) and its probability
    # (also maxlikelihood of trial's data).
    n_stim_mean = proba_estimate.shape[1]
    conditions_loc = np.arange(0, n_stim_mean, 1)
    estimate_loc = estimate.values - 1
    proba_data = proba_estimate[
        estimate_loc, conditions_loc
    ]

    # sanity checks
    if (proba_data <= 0).any():
        raise ValueError("""likelihood<0, but must be>0""")
    elif (~np.isreal(proba_data)).any():
        raise ValueError(
            """likelihood is a complex nb.
            It should be Real"""
        )

    return proba_data


def get_proba_percept(
    stim_mean,
    params,
    k_llh,
    k_prior,
    k_card,
    prior_tail,
    p_rand,
):

    # get fixed parameters
    # ....................
    # stimulus
    stim_std = params["task"]["fixed_params"]["stim_std"]
    prior_shape = params["model"]["fixed_params"][
        "prior_shape"
    ]

    # prior
    prior_mode = params["model"]["fixed_params"][
        "prior_mode"
    ]
    prior_std = params["task"]["fixed_params"]["prior_std"]

    # track trials
    n_trials = len(stim_std)

    # percept readout
    readout = params["model"]["fixed_params"]["readout"]

    # sort stimulus & prior std
    stim_std_set = sorted(np.unique(stim_std), reverse=True)
    prior_std_set = sorted(
        np.unique(prior_std), reverse=True
    )

    # get set of task parameters
    stim_mean_set = np.unique(stim_mean)
    n_stim_std = len(stim_std_set)
    n_prior_std = len(prior_std_set)

    # boolean matrix to locate stim std conditions
    # each column of LLHs is mapped to a
    # stim_std_set
    LLHs = np.zeros((n_trials, n_stim_std)) * np.nan
    for ix in range(n_stim_std):
        LLHs[:, ix] = stim_std == stim_std_set[ix]

    # boolean matrix to locate stim std conditions
    Prior = np.zeros((n_trials, n_prior_std)) * np.nan
    for ix in range(n_prior_std):
        Prior[:, ix] = prior_std == prior_std_set[ix]

    # set percept space
    percept_space = np.arange(1, 361, 1)

    # init outputs
    percept_llh = defaultdict(dict)

    # compute MAP percept density
    # over prior and stimlus noises
    for ix in range(len(prior_std_set)):
        for jx in range(n_stim_std):
            (
                readout_percept,
                percept_llh[ix][jx],
            ) = get_bayes_lookup(
                percept_space,
                stim_mean_set,
                k_llh[jx],
                prior_mode,
                k_prior[ix],
                prior_shape,
                k_card,
                readout=readout,
            )

    # now get matrix 'PupoGivenBI' of likelihood values
    # (upos=1:1:360,trials) for possible values of upo
    # (rows) for each trial (column)
    PupoGivenBI = (
        np.zeros((len(readout_percept), len(stim_mean)))
        * np.nan
    )

    # record conditions
    conditions = np.zeros((len(stim_mean), 3)) * np.nan

    # compute PupoGivenBI for each condition
    # in columns
    for s_mean_i in range(len(stim_mean_set)):
        thisd = stim_mean == stim_mean_set[s_mean_i]
        for p_i in range(n_prior_std):
            for s_noise_i in range(len(stim_std_set)):

                # locate this condition's trials
                # for PupoGivenBI
                s_mean_i_bool = thisd.values.astype(bool)
                s_noise_i_bool = LLHs[:, s_noise_i].astype(
                    bool
                )
                p_i_bool = Prior[:, p_i].astype(bool)
                loc_conditions = (
                    s_mean_i_bool
                    & s_noise_i_bool
                    & p_i_bool
                )
                n_cond_repeat = sum(loc_conditions)

                # locate trials for stimulus means
                # in percept_llh
                stim_mean_loc = (
                    stim_mean_set[
                        np.tile(s_mean_i, n_cond_repeat)
                    ]
                    - 1
                )

                # fill in
                PupoGivenBI[
                    :, loc_conditions
                ] = percept_llh[p_i][s_noise_i][
                    :, stim_mean_loc
                ]

                # record conditions
                conditions[
                    loc_conditions, 0
                ] = prior_std_set[p_i]
                conditions[
                    loc_conditions, 1
                ] = stim_std_set[s_noise_i]
                conditions[
                    loc_conditions, 2
                ] = stim_mean_set[s_mean_i]

    # normalize to probabilities
    PupoGivenBI = PupoGivenBI / sum(PupoGivenBI)[None, :]

    # probabilities of percepts "upo" given random estimation
    PupoGivenRand = np.ones((360, n_trials)) / 360

    # calculate probability of percepts "upo" given the model
    PBI = 1 - p_rand
    PupoGivenModel = (
        PupoGivenBI * PBI + PupoGivenRand * p_rand
    )

    # sanity check that proba_percept are probabilitoes
    if not all(sum(PupoGivenModel)) == 1:
        raise ValueError(
            """PupoGivenModel should sum to 1"""
        )

    return {
        "readout_percept": readout_percept,
        "PupoGivenModel": PupoGivenModel,
        "conditions": conditions,
    }


def get_proba_estimate(
    k_m: float, PupoGivenModel: np.ndarray
):

    # convolve percept density with motor noise
    # Now we shortly replace upo=1:1:360 by upo=0:1:359 because motor noise
    # distribution need to peak at 0 and vmPdfs function needs 'x' to contain
    # the mean '0' to work. Then we set back upo to its initial value. This have
    # no effect on the calculations.
    percept_space = np.arange(0, 360, 1)
    motor_mean = np.array([0])
    proba_motor = VonMises(p=True).get(
        percept_space, motor_mean, [k_m]
    )
    n_stim_mean = PupoGivenModel.shape[1]
    proba_motor = np.tile(proba_motor, n_stim_mean)
    PestimateGivenModel = get_circ_conv(
        PupoGivenModel, proba_motor
    )
    # check that probability of estimates Given Model are positive values.
    # circular convolution sometimes produces negative values very close to zero
    # (order of -10^-18). Those values produce infinite -log likelihood which are
    # only need a single error in estimation to be rejected during model fitting
    # (one trial has +inf -log likelihood to be predicted by the model).
    # This is obviously too conservative. We need a minimal non-zero lapse rate.
    # Prandom that allows for errors in estimation. So we add 10^320 the minimal
    # natural number available in matlab. This means that every time an estimate
    # that is never produced by the model without lapse rate is encountered
    # -loglikelihood increases by -log(10^-320) = 737 and is thus less likely to
    # be a good model (lowest -logLLH). But the model is not rejected altogether
    # (it would be -log(0) = inf). In the end models that cannot account for
    # error in estimates are more likely to be rejected than models who can.
    if not is_empty(np.where(PestimateGivenModel <= 0)[0]):
        PestimateGivenModel[
            PestimateGivenModel <= 0
        ] = 10e-320

    # normalize to probabilities
    PestimateGivenModel = (
        PestimateGivenModel
        / sum(PestimateGivenModel)[None, :]
    )
    return PestimateGivenModel


def get_bayes_lookup(
    percept_space: np.array,
    stim_mean: np.array,
    k_llh: float,
    prior_mode: float,
    k_prior: float,
    prior_shape: str,
    k_card: float,
    readout: str,
):
    """Create a bayes lookup matrix
    based on Girshick's paper
    rows: M measurements
    cols: N stimulus feature means
    value: log(likelihood) of percept

    Usage:
        .. code-block::
        
            percept, logl_percept = get_bayes_lookup(
                percept_space=1:1:360,
                stim_mean=5:10:355,
                k_llh=5,
                prior_mode=225,
                k_prior=4.77,
                k_card=0,
                prior_tail=0,
                prior_shape='von_mises',
                )

    Returns:
        (np.array): percepts
        (np.array): percept likelihood

    [TODO]: clarify the conceptual objects used:
        stimulus space, percept space ... 
    """

    # set stimulus feature mean space s (the
    # discrete circular space with unit 1).
    # e.g., feature could be motion direction
    # s_i are each stimulus feature mean
    stim_mean_space = np.arange(1, 361, 1)

    # cast prior mode as an array
    prior_mode = np.array([prior_mode])

    # calculate measurement densities ~v(s_i,k_llh)
    # (m_i x s_i)
    # m_i is measurement i. percept space
    # is the same as the measurement space
    meas_density = VonMises(p=True).get(
        percept_space, stim_mean, [k_llh]
    )

    # calculate likelihood densities
    # (s space x m_i)
    llh = VonMises(p=True).get(
        stim_mean_space, percept_space, [k_llh]
    )

    # calculate learnt prior densities
    # (s space  x m_i)
    learnt_prior = get_learnt_prior(
        percept_space,
        prior_mode,
        k_prior,
        prior_shape,
        stim_mean_space,
    )

    # calculate cardinal prior if chosen
    vm_means = np.array([90, 180, 270, 360])
    mixt_coeff = 0.25
    cardinal_prior = VonMisesMixture(p=True).get(
        percept_space, vm_means, [k_card], mixt_coeff
    )

    # calculate posterior densities
    # (s space  x m_i)
    posterior = do_bayes_inference(
        k_llh,
        prior_mode,
        k_prior,
        stim_mean_space,
        llh,
        learnt_prior,
        cardinal_prior,
    )

    # choose percepts
    # (m_i x p_i)
    percept, max_nb_percept = choose_percept(
        readout, stim_mean_space, posterior
    )

    # get each percept likelihood
    # for each measurement m_i
    percept, percept_likelihood = get_percept_likelihood(
        percept_space,
        stim_mean,
        stim_mean_space,
        meas_density,
        percept,
        max_nb_percept,
    )
    return percept, percept_likelihood


def get_percept_likelihood(
    percept_space,
    stim_mean,
    stim_mean_space,
    meas_density,
    percept,
    max_nb_percept,
):
    """calculate percepts' likelihood.
    
    It is the P(m_i|s_i) of the m_i that produced that data
    map m_i and percept(s) p_i in P(p|m_i)
    P(p|m_i) is important because e.g., when
    a m_i produces a bimodal posterior in response to
    a stimulus (e.g., {flat likelihood, bimodal prior}),
    the two modes are equally likely P(p_1|m_i)
    = P(p_2|mi) = 0.5 and P(p_1|s_i) =
    P(p_1|s_i) = P(p_1|m_i)*P(m_i|s_i)
    = 0.5*P(m_i|s_i). Similarly, P(p_2|s_i) =
    P(p_2|m_i)*P(m_i|s_i) = 0.5*P(m_i|s_i).
    The P(p|m_i) of each percept observed for a
    m_i is 1/len(p).
    percepts only depend on m_i which determines likelihood.
    The displayed stimulus determines percept probability.
    Percepts per m_i (rows) are the same across stimulus
    feature means (cols). m_i rows are repeated in the matrix
    when a m_i produces many percepts.
    e.g., the matrices for a max number of percept per m_i=2
        
    Args:
        percept_space (_type_): _description_
        stim_mean (_type_): _description_
        stim_mean_space (_type_): _description_
        meas_density (_type_): _description_
        percept (_type_): _description_
        max_nb_percept (_type_): _description_

    Returns:
        _type_: _description_
    """

    # count percepts by m_i
    # (m_i x 0)
    max_nb_pi_given_mi = np.sum(~np.isnan(percept), 1)

    # probability of percepts given m_i
    # p(p_i|m_i)
    # (m_i x 1)
    prob_pi_given_mi = 1 / max_nb_pi_given_mi

    # assign equiprobability to percepts with
    # same m_i (across cols)
    # (m_i x p_i)
    prob_pi_given_mi = np.tile(
        prob_pi_given_mi[:, None], max_nb_percept
    )

    # reshape as column vector (column-major)
    # (m_i*p_i x s_i)
    prob_pi_given_mi = prob_pi_given_mi.flatten("F")
    prob_pi_given_mi = np.tile(
        prob_pi_given_mi[:, None], len(stim_mean)
    )

    # associate P(m_i|s_i) of each m_i (row)
    # for each stimulus feature mean s_i(col)
    # e.g., the matrices for a max number of
    # percept per m_i = 2
    #
    #    mPdfs_percept1 . . dir_D
    #        .
    #        .
    #       mi_M
    #    mPdfs_percept2 . . dir_D
    #        .
    #        .
    #       mi_M
    # (x si)
    prob_mi_given_si = np.tile(
        meas_density, (max_nb_percept, 1)
    )

    # reshape as column vector (column-major)
    flatten_percept = percept.flatten("F")[:, None]

    # create a percept space
    # to map with likelihood values
    u_l = np.tile(percept_space[:, None], max_nb_percept)

    # reshape as column vector (column-major)
    u_l = u_l.flatten("F")[:, None]

    # map percept and its likelihood
    # to sort by percept
    to_sort = np.hstack(
        [
            u_l,
            flatten_percept,
            prob_pi_given_mi,
            prob_mi_given_si,
        ]
    )

    # sort by percept
    to_sort = to_sort[to_sort[:, 1].argsort()]

    # unpack
    u_l = to_sort[:, 0][:, None]
    percept = to_sort[:, 1][:, None]
    prob_pi_given_mi = to_sort[:, 2 : 2 + len(stim_mean)]
    prob_mi_given_si = to_sort[:, 2 + len(stim_mean) :]

    # likelihood of percepts given stimulus
    # p(p_i|s_i)
    prob_pi_given_si = prob_pi_given_mi * prob_mi_given_si

    # Set the likelihood=0 for percepts not produced
    # because the model cannot produce those percepts
    # even at a reasonable resolution of stimulus feature
    # mean
    percept_set = np.unique(percept[~np.isnan(percept)])
    missing_pi = np.array(
        tuple(set(percept_space) - set(percept_set))
    )[:, None]
    nb_missing_pi = len(missing_pi)

    # Add unproduced percept and set their
    # likelihood=0, then re-sort by percept
    prob_missing_pi_given_si = np.zeros(
        (nb_missing_pi, len(stim_mean))
    )
    prob_all_pi_given_si = np.vstack(
        [prob_pi_given_si, prob_missing_pi_given_si]
    )
    all_pi = np.vstack([percept, missing_pi])
    missing_u_l = np.zeros((nb_missing_pi, 1)) * np.nan
    u_l = np.vstack([u_l, missing_u_l])

    # map all objects to sort them
    to_sort = np.hstack([u_l, all_pi, prob_all_pi_given_si])

    # sort by percept
    to_sort = to_sort[to_sort[:, 1].argsort()]
    all_pi = to_sort[:, 1]
    prob_all_pi_given_si = to_sort[:, 2:]

    # likelihood of each percept (rows are percepts, cols are motion directions,
    # values are likelihood). When a same percept has been produced by different
    # mi produced by the same motion direction, then the percept's likelihood is
    # its mi likelihood.  The likelihoods are properly scaled at the end to sum
    # to 1. e.g.,
    # ................................................................................
    #    if only mi_1 produces?percept=100? and mi_2 also produces?percept=100?
    #  ?and mi_1 and mi_2 are both produced by the same motion direction di
    #    P(percept|di) = P(percept|mi_1)*P(mi_1|di) + P(percept|mi_2)*P(mi_2|dir)
    #    P(percept|mi_1) = P(percept|mi_1) = 1 because both mi only produce one percept
    #    (the same)
    #    so P(percept|di) = P(mi_1|di) + P(mi_2|dir)
    # ................................................................................
    #
    # note: we can see horizontal stripes of 0 likelihood near the obliques when
    # cardinal prior is strong because obliques percepts are never produced.
    # The range of percepts not produced near the obliques increase significantly
    # with cardinal prior strength.
    percept = np.unique(all_pi[~np.isnan(all_pi)])
    prob_pi_set_given_si = (
        np.zeros([len(percept), len(stim_mean)]) * np.nan
    )

    # find measurements that produced this same percept
    # and average probabilities over evidences mi that
    # produces this same percept
    for ix in range(len(percept)):
        loc_pi_set = all_pi == percept[ix]
        prob_pi_set_given_si[ix, :] = sum(
            prob_all_pi_given_si[loc_pi_set, :], 0
        )
    prob_pi_set_given_si[np.isnan(prob_pi_set_given_si)] = 0

    # calculate likelihood of each
    # unique percept
    # matrix of percepts (rows) x
    # 360 stimulus mean space (cols)
    percept_likelihood = (
        np.zeros((len(percept), len(stim_mean_space)))
        * np.nan
    )
    stim_mean_loc = stim_mean - 1
    percept_likelihood[
        :, stim_mean_loc
    ] = prob_pi_set_given_si

    # normalize to probability
    percept_likelihood = (
        percept_likelihood
        / sum(percept_likelihood)[None, :]
    )
    return percept, percept_likelihood


def choose_percept(
    readout: str,
    stim_mean_space: np.ndarray,
    posterior: np.ndarray,
):
    """choose percept(s) for each
    measurement m_i produced by a stimulus
    s_i

    Args:
        readout (str): the posterior readout
        - 'map': maximum a posteriori decision
        stim_mean_space (np.ndarray): the space
            of stimulus feature mean (e.g., motion 
            directions)
        posterior (np.ndarray): the posterior 

    Raises:
        ValueError: _description_
        NotImplementedError: _description_
        ValueError: _description_

    Returns:
        (np.ndarray): percept 
        (int): max_n_percept 
        ul
    """

    # choose a percept p_i (readout)
    # (s_i x p_i)
    # the measurement space is assumed to
    # be the same as the stimulus feature space
    meas_space_size = len(stim_mean_space)
    stim_space_size = len(stim_mean_space)
    percept = (
        np.zeros((stim_space_size, meas_space_size))
        * np.nan
    )

    # when the readout is maximum-a-posteriori
    if readout == "map":

        # find the maximum-a-posteriori estimate(s)(maps)
        # mapped with each m_i (rows). A m_i can produce
        # many maps (many cols for a row) .e.g., when both
        # likelihood and learnt prior are weak, an evidence
        # produces a relatively flat posterior which maximum
        # can not be estimated accurately. max(posterior)
        # produces many maps with equal probabilities.

        # map each m_i with its percept(s)
        for meas_i in range(meas_space_size):

            # locate each posterior's maximum
            # a posteriori(s)
            loc_percept = posterior[:, meas_i] == np.max(
                posterior[:, meas_i]
            )

            # count number of percepts
            n_percepts = sum(loc_percept)

            # map measurements (rows)
            # with their percepts (cols)
            percept[meas_i, :n_percepts] = stim_mean_space[
                loc_percept
            ]

            # handle exception
            # check that all measurements have at
            # least a percept
            if n_percepts == 0:
                raise ValueError(
                    f"""Measurement {meas_i}
                    has no percept(s)."""
                )
    else:
        # handle exception
        raise NotImplementedError(
            f"""
            Readout {readout} has not 
            yet been implemented. 
            """
        )

    # replace 0 by 360 degree
    percept[percept == 0] = 360

    # handle exception
    is_percept = percept[~np.isnan(percept)]
    if (is_percept > 360).any() or (is_percept < 1).any():
        raise ValueError(
            """Percepts must belong to [1,360]."""
        )

    # drop nan percepts from cols
    # (m_i x max_nb_percept)
    max_nb_percept = max(np.sum(~np.isnan(percept), 1))
    percept = percept[:, :max_nb_percept]
    return percept, max_nb_percept


def get_learnt_prior(
    percept_space: np.ndarray,
    prior_mode: np.ndarray,
    k_prior: float,
    prior_shape: str,
    stim_mean_space,
):
    """calculate the learnt prior probability 
    distribution
    cols: prior for each m_i, are the same
    rows: stimulus feature mean space (e.g.,
    motion direction) 

    Args:
        percept_space (np.ndarray): _description_
        prior_mode (np.ndarray): _description_
        k_prior (float): _description_
        prior_shape (str): shape of the prior
        - 'vonMisesPrior'
        stim_mean_space (np.ndarray): stimulus 
        feature mean space: (1:1:360)

    Returns:
        (np.ndarray): matrix of learnt priors
        rows: stimulus feature mean space
        cols: m_i
    """
    if prior_shape == "vonMisesPrior":
        # create prior density
        # (Nstim mean x 1)
        learnt_prior = VonMises(p=True).get(
            stim_mean_space, prior_mode, [k_prior],
        )
        # repeat the prior across cols
        # (Nstim mean x Nm_i)
        learnt_prior = np.tile(
            learnt_prior, len(percept_space)
        )
    return learnt_prior


def do_bayes_inference(
    k_llh,
    prior_mode,
    k_prior,
    stim_mean_space,
    llh: np.ndarray,
    learnt_prior: np.ndarray,
    cardinal_prior: np.ndarray,
):

    """Realize Bayesian inference    
    """

    # do Bayesian integration
    posterior = llh * learnt_prior * cardinal_prior

    # normalize columns to sum to 1
    # for small to intermediate values of k
    loc = np.where(np.isnan(posterior[0, :]))[0]
    posterior[:, ~loc] = (
        posterior[:, ~loc]
        / sum(posterior[:, ~loc])[None, :]
    )

    # round posteriors
    # We fix probabilities at 10e-6 floating points
    # We can get posterior modes despite round-off errors
    # We tried with different combinations of 10^6 and
    # round instead of fix. If we don't round enough we
    # cannot get the modes of the posterior accurately
    # because of round-off errors. If we round too much
    # we get more modes than we should, but the values
    # obtained are near the true modes so I choose to round
    # more (same as in simulations).
    # sum over rows
    posterior = np.round(posterior, 6)

    # TRICK: When k gets very large, e.g., for the prior,
    # most values of the prior becomes 0 except close to
    # the mean. The product of the likelihood and prior
    # only produces 0 values for all directions, particularly
    # as motion direction is far from the prior. Marginalization
    # (scaling) makes them NaN. If we don't correct for that,
    # fit is impossible. In reality a von Mises density will
    # never have a zero value at its tails. We use the closed-from
    # equation derived by Murray and Morgenstern, 2010.
    if not is_empty(loc):
        # use Murray and Morgenstern., 2010
        # closed-form equation
        # mode of posterior
        m_i = stim_mean_space[loc]
        mirad = get_deg_to_rad(m_i, True)
        uprad = get_deg_to_rad(prior_mode, True)

        # set k ratio
        k_ratio = k_llh / k_prior
        if k_llh == np.inf and k_prior == np.inf:
            k_ratio = 1

        # calculate posterior's mean
        upo = np.round(
            mirad
            + arctan2(
                sin(uprad - mirad),
                k_ratio + cos(uprad - mirad),
            )
        )
        # make sure upo belongs to stimulus
        # mean space
        upo = np.round(get_rad_to_deg(upo))

        # when prior and likelihood strengths are
        # infinite
        if k_llh == np.inf or k_prior == np.inf:
            kpo = np.inf
        else:
            # else use Morgenstern equation to
            # calculate posteriors' strengths
            kpo = np.sqrt(
                k_llh ** 2
                + k_prior ** 2
                + 2 * k_prior * k_llh * cos(uprad - mirad)
            )
            kpo = kpo.squeeze().tolist()

        # create those posteriors
        posterior[:, loc] = VonMises(p=True).get(
            stim_mean_space, upo, kpo,
        )

    return posterior


def predict(
    fit_p: np.ndarray,
    params: Dict[str, any],
    stim_mean: pd.Series,
    data: pd.Series,
    granularity: str,
):
    """""get model predictions

    Args:
        fit_p (np.ndarray): model free parameters
        params (Dict[str, any]): model and task 
        fixed parameters
        stim_mean (pd.Series): _description_
        data (pd.Series): _description_
        granularity (str): 
        - "trial"
        - "mean"
        
    Returns:
        (dict): _description_
    """

    # get best fit's calculated variables
    _, output = get_fit_variables(
        fit_p, params, stim_mean, data=data
    )

    # get prediction stats
    if granularity == "mean":
        output = get_prediction_stats(output)

    # predict per trial
    elif granularity == "trial":
        output = get_prediction_stats(output)
        output = get_trial_prediction(
            output, params, n_repeats=5
        )
    else:
        raise ValueError(
            """granularity only takes "mean" or "trial" args"""
        )
    return output


def simulate_dataset(
    fit_p: np.ndarray,
    params: Dict[str, any],
    stim_mean: pd.Series,
    granularity: str,
    **kwargs: dict,
):
    """""get model predictions

    Args:
        fit_p (np.ndarray): model free parameters
        params (Dict[str, any]): model and task 
        fixed parameters
        stim_mean (pd.Series): _description_
        stim_estimate (pd.Series): _description_
        granularity (str): 
        - "trial": prediction are stochastic choices sampled 
        from the model's generative probability density
        - "mean": prediction statistics (mean and std calculated
        from the model's generative probability density)
    
    kwargs:
        when granularity="trial":
        - n_repeats (int): the number of repeats of 
        each task condition
        

    Returns:
        (dict): _description_

    [TODO]: drop "stim_estimate" from Args
    """

    # get best fit's calculated variables
    _, output = get_fit_variables(fit_p, params, stim_mean)

    # get prediction statistics or
    # as stochastic choices sampled
    # from the model's generative probability
    # density
    if granularity == "mean":
        output = get_prediction_stats(output)
    elif granularity == "trial":
        output = get_prediction_stats(output)
        output = get_trial_prediction(
            output, params, n_repeats=kwargs["n_repeats"]
        )
    else:
        raise ValueError(
            """granularity only takes "mean" or "trial" args"""
        )
    return output


def get_trial_prediction(
    output: dict, params: dict, n_repeats: int
):
    """_summary_

    Args:
        output (dict): _description_
        params (dict): _description_
        n_repeats (int): _description_

    Returns:
        (dict): _description_
        - output["dataset"]: for now "stim_mean" must be int
    """

    # get task conditions by trial
    prior_mode = params["model"]["fixed_params"][
        "prior_mode"
    ]
    prior_shape = params["model"]["fixed_params"][
        "prior_shape"
    ]

    # get conditions
    # (N_conditions x 3 params)
    cond = output["conditions"]

    # initialize dataset
    data_space = np.arange(0, 360, 1)
    stim_mean = []
    stim_std = []
    prior_mode_tmp = []
    prior_std = []
    prior_shape_tmp = []
    estimate = []

    # loop over conditions
    for c_i in range(cond.shape[0]):

        # sample n_repeats trial predictions
        data = np.random.choice(
            data_space,
            size=n_repeats,
            p=output["PestimateGivenModel"][:, c_i],
            replace=True,
        )

        # build dataset
        stim_mean += np.tile(
            cond[c_i, 2], n_repeats
        ).tolist()
        stim_std += np.tile(
            cond[c_i, 1], n_repeats
        ).tolist()
        prior_mode_tmp += np.tile(
            prior_mode, n_repeats
        ).tolist()
        prior_std += np.tile(
            cond[c_i, 0], n_repeats
        ).tolist()
        prior_shape_tmp += np.tile(
            prior_shape, n_repeats
        ).tolist()
        estimate += data.tolist()

    # record dataset
    output["dataset"] = pd.DataFrame()
    output["dataset"]["stim_mean"] = stim_mean
    output["dataset"]["stim_std"] = stim_std
    output["dataset"]["prior_mode"] = prior_mode_tmp
    output["dataset"]["prior_std"] = prior_std
    output["dataset"]["prior_shape"] = prior_shape_tmp
    output["dataset"]["estimate"] = estimate

    # always cast stimulus mean as integer
    # [TODO]: enable floats
    output["dataset"]["stim_mean"] = output["dataset"][
        "stim_mean"
    ].astype(int)

    # name parameters
    output["dataset"].columns = [
        "stim_mean",
        "stim_std",
        "prior_mode",
        "prior_std",
        "prior_shape",
        "estimate",
    ]

    return output


def get_prediction_stats(output: dict) -> dict:
    """calculate prediction statistics

    Args:
        output (dict): _description_

    Returns:
        dict: _description_
    """

    # extract fit variables
    proba_estimate = output["PestimateGivenModel"]
    map = output["map"].copy()

    # get conditions
    cond = output["conditions"]

    # initialise stats
    prediction_mean = []
    prediction_std = []

    # get set of conditions
    # (N_conditions x 3 task params)
    cond_set, cond_set_ix, _ = get_combination_set(cond)
    proba_estimate = proba_estimate[:, cond_set_ix]

    # record statistics by condition
    for c_i in range(len(cond_set)):
        pred = get_circ_weighted_mean_std(
            map, proba_estimate[:, c_i], type="polar",
        )
        prediction_mean.append(pred["deg_mean"])
        prediction_std.append(pred["deg_std"])

    # record predictions stats
    output["prediction_mean"] = np.array(prediction_mean)
    output["prediction_std"] = np.array(prediction_std)
    return output


def get_data_stats(data: np.ndarray, output: dict):
    """calculate the data statistics 

    Args:
        data (np.ndarray): _description_
        output (dict): _description_

    Returns:
        (dict): data statistics
    """
    # get conditions
    cond = output["conditions"]

    # initialise statistics
    data_mean = []
    data_std = []

    # get set of conditions
    cond_set, ix, _ = get_combination_set(cond)

    # record stats by condition
    for c_i in range(len(cond_set)):

        # find condition's instances
        loc_1 = cond[:, 0] == cond_set[c_i, 0]
        loc_2 = cond[:, 1] == cond_set[c_i, 1]
        loc_3 = cond[:, 2] == cond_set[c_i, 2]

        # get associated data
        data_c_i = data.values[loc_1 & loc_2 & loc_3]

        # set each instance with equal probability
        trial_proba = np.tile(
            1 / len(data_c_i), len(data_c_i)
        )

        # get statistics
        stats = get_circ_weighted_mean_std(
            data_c_i, trial_proba, type="polar",
        )

        # record statistics
        data_mean.append(stats["deg_mean"])
        data_std.append(stats["deg_std"])

    # record statistics
    output["data_mean"] = np.array(data_mean)
    output["data_std"] = np.array(data_std)

    # record their condition
    output["conditions"] = cond_set
    return output


def get_combination_set(database: np.ndarray):
    """Get set of combinations

    Args:
        database (np.ndarray): _description_

    Returns:
        _type_: _description_
        combs is the set of combinations
        b are the row indices for each combination in database
        c are the rows indices for each combination in combs
    """
    combs, ia, ic = np.unique(
        database,
        return_index=True,
        return_inverse=True,
        axis=0,
    )
    return combs, ia, ic

