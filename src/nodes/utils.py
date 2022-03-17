"""
    module

    Copyright 2022 by Steeve Laquitaine, GNU license 
"""

from collections import defaultdict

import numpy as np
import pandas as pd
from numpy import arctan2, cos, sin
from scipy.optimize import fmin
from src.nodes.data import VonMises
from src.nodes.util import (get_circ_conv, get_deg_to_rad, get_rad_to_deg,
                            is_empty)

pd.options.mode.chained_assignment = None


def fit_maxlogl(
    data, prior_shape: str, prior_mode: float, readout: str
):
    """Fits estimate data with the
    maximum log(likelihood) method
    This method searches for the model parameters
    that maximize the log(likeligood) of the data

    Args:
        data (pd.DataFrame): _description_  
        prior_shape (str): shape of the prior  
        - "vonMisesPrior"  
        prior_mode: (float): mode of the prior  

    Returns:
        _type_: _description_
    """
    # set 0 to 360 deg
    loc_zero = data["estimate"] == 0
    data["estimate"][loc_zero] = 360

    # initialize model params
    K_LLH = [1]
    K_PRIOR = [1]
    K_CARD = [1]
    PRIOR_TAIL = [0]
    PRAND = [0]

    # store model params
    model_params = [
        K_LLH,
        K_PRIOR,
        K_CARD,
        PRIOR_TAIL,
        PRAND,
    ]

    # set model architecture
    model = (readout,)

    # store task params
    task_params = (
        (
            data["stim_std"],
            prior_mode,
            data["prior_std"],
            prior_shape,
        ),
    )

    # set data
    stim_mean = data["stim_mean"]
    estimate = data["estimate"]

    # train model
    output = fmin(
        get_logl,
        model_params,
        args=(stim_mean, estimate, task_params, model),
        disp=True,
        retall=True,  # get solutions after iter
        maxiter=100,  # max nb of iter
        maxfun=100,  # max nb of func eval
    )

    # get results
    best_fit_params = output[0]
    neglogl = output[1]
    return {
        "neglogl": neglogl,
        "best_fit_params": best_fit_params,
    }


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
    stim_mean: pd.Series,
    estimate: pd.Series,
    task_params: tuple,
    model: tuple,
):
    """_summary_

    Args:
        model_params (list): _description_
        stim_mean (pd.Series): _description_
        estimate (pd.Series): _description_
        task_params (tuple): _description_

    Returns:
        _type_: _description_
    """

    # get task params
    (
        stim_std,
        prior_mode,
        prior_std,
        prior_shape,
    ) = task_params[0]

    # sort stim and prior std
    stim_std_set = sorted(np.unique(stim_std), reverse=True)
    prior_std_set = sorted(
        np.unique(prior_std), reverse=True
    )

    # get unique task params
    stim_mean_set = np.unique(stim_mean)
    n_stim_std = len(stim_std_set)
    n_prior_std = len(prior_std_set)

    # get number of free parameters
    # to train
    n_fit_params = sum(~np.isnan(model_params))

    # get model params
    model_params = model_params.tolist()
    k_llh = model_params[:n_stim_std]
    del model_params[:n_stim_std]
    k_prior = model_params[:n_prior_std]
    del model_params[:n_prior_std]
    k_cardinal = model_params.pop(0)
    prior_tail = model_params.pop(0)
    p_rand = model_params[0]

    # boolean matrix to locate stim std conditions
    # each column of LLHs is mapped to a
    # stim_std_set
    LLHs = (
        np.zeros((len(stim_std), len(stim_std_set)))
        * np.nan
    )
    for i in range(len(stim_std_set)):
        LLHs[:, i] = stim_std == stim_std_set[i]

    # boolean matrix to locate stim std conditions
    Prior = np.zeros((len(prior_std), n_prior_std)) * np.nan
    for i in range(n_prior_std):
        Prior[:, i] = prior_std == prior_std_set[i]

    # set percept space
    percept_space = np.arange(1, 361, 1)

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
                readout=model[0],
            )

    # now get matrix 'PupoGivenBI' of likelihood values
    # (upos=1:1:360,trials) for possible values of upo
    # (rows) for each trial (column)
    PupoGivenBI = (
        np.zeros((len(map), len(estimate))) * np.nan
    )
    for i in range(len(stim_mean_set)):
        # displayed direction
        thisd = stim_mean == stim_mean_set[i]
        for j in range(n_prior_std):
            for k in range(len(stim_std_set)):
                condition_loc = (
                    thisd & LLHs[:, k] & Prior[:, j]
                )
                n_cond_repeat = sum(condition_loc)
                stim_mean_loc = (
                    stim_mean_set[np.tile(i, n_cond_repeat)]
                    - 1
                )
                PupoGivenBI[:, condition_loc] = llh_map[j][
                    k
                ][:, stim_mean_loc]

    # normalize to probabilities
    PupoGivenBI = PupoGivenBI / sum(PupoGivenBI)[None, :]

    # probabilities of percepts "upo" given random estimation
    PupoGivenRand = np.ones((360, len(stim_mean))) / 360

    # calculate probability of percepts "upo" given the model
    PBI = 1 - p_rand
    PupoGivenModel = (
        PupoGivenBI * PBI + PupoGivenRand * p_rand
    )

    # check PupoGivenModel sum to 1
    if not all(sum(PupoGivenModel)) == 1:
        raise ValueError("PupoGivenModel should sum to 1")

    # convolve with motor noise
    # -------------------------
    # Now we shortly replace upo=1:1:360 by upo=0:1:359 because motor noise
    # distribution need to peak at 0 and vmPdfs function needs 'x' to contain
    # the mean '0' to work. Then we set back upo to its initial value. This have
    # no effect on the calculations.
    upo = np.arange(1, 361, 1)
    stuff = np.array([360])
    Pmot = VonMises(p=True).get(upo, stuff, k_llh)
    Pmot_to_conv = np.tile(Pmot, len(stim_mean))
    PestimateGivenModel = get_circ_conv(
        PupoGivenModel, Pmot_to_conv
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

    # set upo to initial values any case we use it later
    upo = np.arange(1, 361, 1)

    # normalize to probabilities
    PestimateGivenModel = (
        PestimateGivenModel
        / sum(PestimateGivenModel)[None, :]
    )

    # get the log likelihood of the observed estimates
    # other case are when we just want estimates
    # distributions prediction given
    # model parameters
    estimate[estimate[estimate == 0]] = 360

    # single trial's measurement, its position(row)
    # for each trial(col) and its probability
    # (also maxlikelihood of trial's data).
    # make sure sub2ind inputs are the same size
    conditions_loc = np.arange(0, len(stim_mean), 1)
    estimate_loc = estimate.values - 1
    PdataGivenModel = PestimateGivenModel[
        estimate_loc, conditions_loc
    ]

    # sanity checks
    if (PdataGivenModel <= 0).any():
        raise ValueError("""likelihood<0, but must be>0""")
    elif (~np.isreal(PdataGivenModel)).any():
        raise ValueError(
            """likelihood is complex. 
            It should be Real"""
        )

    # We use log likelihood because
    # likelihood is so small that matlab cannot
    # encode it properly (numerical unstability).
    # We can use single trials log
    # likelihood to calculate AIC in the conditions
    # that maximize differences in
    # predictions of two models.
    Logl_pertrial = np.log(PdataGivenModel)

    # we minimize the objective function
    # -sum(log(likelihood))
    negLogl = -sum(Logl_pertrial)

    # akaike information criterion metric
    aic = 2 * (n_fit_params - sum(Logl_pertrial))

    # print
    print(
        f"""-logl: {negLogl}, aic: {aic}, 
                             k_llh: {k_llh}, 
                             k_prior: {k_prior}, 
                             k_card: {k_cardinal}, 
                             pr_tail: {prior_tail}, 
                             p_rnd: {p_rand}"""
    )
    return negLogl


def get_bayes_lookup(
    percept_space: np.array,
    stim_mean: np.array,
    k_llh: float,
    prior_mode: float,
    k_prior: float,
    k_cardinal: float,
    prior_tail: float,
    prior_shape: str,
    readout: str,
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

    # set stimulus feature space (the
    # entire discretized circular space with unitary
    # unit). An example feature could be an object's
    # motion direction
    stim_mean_space = np.arange(1, 361, 1)

    # cast prior mode as array
    prior_mode = np.array([prior_mode])

    # calculate measurement densities ~v(di,kl)
    # di are stimulus mean in columns
    # over range of possible measurement in rows
    meas_density = VonMises(p=True).get(
        percept_space, stim_mean, [k_llh]
    )

    # calculate likelihood densities
    # likelihood of stimulus feature di (row)
    # given measurements mi (col)
    llh = VonMises(p=True).get(
        stim_mean_space, percept_space, [k_llh]
    )

    # calculate learnt prior densities
    # it is the same for each mi in cols over motion
    # directions in rows
    if prior_shape == "vonMisesPrior":
        learnt_prior = VonMises(p=True).get(
            stim_mean_space, prior_mode, [k_prior],
        )
        learnt_prior = np.tile(
            learnt_prior, len(percept_space)
        )

    # calculate posterior
    posterior = do_bayes_inference(
        k_llh,
        prior_mode,
        k_prior,
        stim_mean_space,
        llh,
        learnt_prior,
    )

    # choose a percept (readout)
    percept = (
        np.zeros((len(stim_mean_space), len(percept_space)))
        * np.nan
    )
    if readout == "map":
        # find the MAPs estimates (values) of each mi (rows).
        # A mi may produce more than one MAP (more than one value for a row).
        # e.g., when both likelihood and learnt prior are weak, an evidence produces
        # a relatively flat posterior which maximum can not be estimated accurately.
        # max(posterior) produces two (or more) MAP values. Those MAP values are
        # observed with the same probability (e.g., 50#  if two MAPs) given a mi.
        for meas_i in range(len(percept)):
            # get each posterior's maximum
            # a posteriori(s)
            map_loc = posterior[:, meas_i] == np.max(
                posterior[:, meas_i]
            )
            n_maps = sum(map_loc)
            percept[meas_i, 0:n_maps] = stim_mean_space[
                map_loc
            ]

            # sanity checks
            # check that all measurements have at
            # least a percept
            if n_maps == 0:
                raise ValueError(
                    """Some measurements 
                    have no percepts."""
                )
    else:
        raise NotImplementedError(
            """
            readout has not yet
            been implemented
            """
        )

    percept[percept == 0] = 360

    # run sanity check
    is_percept = percept[~np.isnan(percept)]
    if (is_percept > 360).any() or (is_percept < 1).any():
        raise ValueError(
            """percept must be >=1 and <=360"""
        )

    # calculate likelihood of each data upo that is the P(mi|motion dir)
    # of the mi that produced the upo
    max_n_percept = max(np.sum(~np.isnan(percept), 0))
    ul = np.tile(percept_space[:, None], max_n_percept)
    ul = ul.flatten()[:, None]

    # map to percept estimates
    percept = percept[:, 0:max_n_percept]

    # associate P(percept|mi)
    # P(percept|mi) is important because e.g., when the same mi produces a
    # posterior with two modes given a motion direction (flat likelihood and bimodal prior),
    # the two modes are observed equiprobably P(percept_1|mi) = P(percept_2|mi) = 0.5 and P(percept_1|di) =
    # P(percept_1|di) = P(percept_1|mi)*P(mi|di) = 0.5*P(mi|di) and in the same way
    # P(percept_2|di) = P(percept_2|mi)*P(mi|di) = 0.5*P(mi|di)
    # The P(percept|mi) of each percept observed for a mi is 1/nb_of_percept
    n_percept_each_mi = np.sum(~np.isnan(percept), 1)

    # percepts value only depends on mi value which determine likelihood position
    # and thus posterior position. The displayed motion directions have not control on
    # percepts values. They only determine the Probability of observing percepts.
    # So the percepts and the number of percepts observed for each mi (row) is the same for
    # all motion directions displayed (col). Row (mi) repetitions appears in the matrix
    # when a mi produced a posterior with two modes.
    # e.g., the matrices for a max number of percept per mi of 2
    #
    #    mPdfs_percept1 . . dir_D
    #        .
    #        .
    #       mi_M
    #    mPdfs_percept2 . . dir_D
    #        .
    #        .
    #       mi_M

    # sanity checks
    # check that all mi have a percept
    # if (n_percept_each_mi == 0).any():
    #     print(n_percept_each_mi)
    #     raise ValueError("n_percept_each_mi==0")

    PperceptgivenMi = 1 / n_percept_each_mi
    PperceptgivenMi = np.tile(
        PperceptgivenMi[:, None], max_n_percept
    )
    PperceptgivenMi = PperceptgivenMi.flatten()
    PperceptgivenMi = np.tile(
        PperceptgivenMi[:, None], len(stim_mean)
    )

    # associate P(mi|di) of each mi (row) for each motion dir di(col)
    # e.g., the matrices for a max number of percept per mi of 2
    #
    #    mPdfs_percept1 . . dir_D
    #        .
    #        .
    #       mi_M
    #    mPdfs_percept2 . . dir_D
    #        .
    #        .
    #       mi_M
    PmiGivenDi = np.tile(meas_density, max_n_percept).T

    flat_percept = percept.flatten()[:, None]
    the_matx = np.hstack(
        [ul, flat_percept, PperceptgivenMi, PmiGivenDi,]
    )

    # sort by percept
    the_matx = the_matx[the_matx[:, 1].argsort()]

    # re-extract
    ul = the_matx[:, 0][:, None]
    percept = the_matx[:, 1][:, None]
    PperceptgivenMi = the_matx[:, 2 : 2 + len(stim_mean)]
    PmiGivenDi = the_matx[:, 2 + len(stim_mean) :]

    # likelihood of each percept estimate
    PperceptgivenDi = PperceptgivenMi * PmiGivenDi

    # Set the likelihood of percepts not produced at 0 because the model cannot
    # produce those estimates even at a reasonable resolutions of motion
    # direction
    uniqpercept = np.unique(percept[~np.isnan(percept)])
    missingpercept = np.array(
        tuple(set(np.arange(1, 361, 1)) - set(uniqpercept))
    )[:, None]
    numMissingpercept = len(missingpercept)

    # Add unproduced percept and set their likelihood=0
    # the re-sort by percept
    PmissingperceptgivenDi = np.zeros(
        (numMissingpercept, len(stim_mean))
    )
    PallperceptgivenDi = np.vstack(
        [PperceptgivenDi, PmissingperceptgivenDi]
    )
    allpercept = np.vstack([percept, missingpercept])
    ulnonExistent = (
        np.zeros((numMissingpercept, 1)) * np.nan
    )
    ul = np.vstack([ul, ulnonExistent])
    TheMatrixII = np.hstack(
        [ul, allpercept, PallperceptgivenDi]
    )
    # sort by percept
    TheMatrixII = TheMatrixII[TheMatrixII[:, 1].argsort()]
    allpercept = TheMatrixII[:, 1]
    PallperceptgivenDi = TheMatrixII[:, 2:]

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
    uniqallpercept = np.unique(
        allpercept[~np.isnan(allpercept)]
    )
    PuniqperceptgivenDi = (
        np.zeros([len(uniqallpercept), len(stim_mean)])
        * np.nan
    )

    # find measurements that produced this same percept
    # and average probabilities over evidences mi that
    # produces this same percept
    for ix in range(len(uniqallpercept)):
        posUniqpercept = allpercept == uniqallpercept[ix]
        PuniqperceptgivenDi[ix, :] = sum(
            PallperceptgivenDi[posUniqpercept, :], 0
        )
    PuniqperceptgivenDi[np.isnan(PuniqperceptgivenDi)] = 0

    # calculate likelihood of each
    # unique percept
    # matrix of percepts (rows) x
    # 360 stimulus mean (cols)
    llh_percept = (
        np.zeros(
            (len(uniqallpercept), len(stim_mean_space))
        )
        * np.nan
    )
    stim_mean_loc = stim_mean - 1
    llh_percept[:, stim_mean_loc] = PuniqperceptgivenDi

    # normalize to probability
    llh_percept = llh_percept / sum(llh_percept)[None, :]
    # from matplotlib import pyplot as plt
    # plt.imshow(llh_percept); plt.show()
    return uniqallpercept, llh_percept


def do_bayes_inference(
    k_llh,
    prior_mode,
    k_prior,
    stim_mean_space,
    llh,
    learnt_prior,
):

    # do Bayesian integration
    posterior = llh * learnt_prior

    # normalize cols to sum to 1
    posterior = posterior / sum(posterior)[None, :]

    # round posterior
    # We fix probabilities at 10e-6 floating points
    # This permits to get the modes of the posterior despite round-off
    # errors. Try with different combinations of 10^6 and round instead of fix
    # If we don't round enough we cannot get the modes of the posterior
    # accurately due to round-off errors. But, now if we
    # round too much we get more modes than we should, but the values
    # obtained surf around the values of the true modes so I choose to round
    # more than not enough (same as in simulations).
    # sum over rows
    posterior = np.round(posterior, 6)

    # TRICK: When k gets very large, e.g., for the prior, most values of the prior
    # becomes 0 except close to the mean. The product of the likelihood and
    # prior only produces 0 values for all directions, particularly as motion
    # direction is far from the prior. Marginalization (scaling) makes them NaN.
    # If we don't correct for that fit is not possible. In reality a von Mises
    # density will never have a zero value at its tails. We use the closed-from
    # equation derived in Murray and Morgenstern, 2010.
    loc = np.where(np.isnan(posterior[0, :]))[0]
    if not is_empty(loc):
        # use Murray and Morgenstern., 2010
        # closed-form equation
        # mode of posterior
        mi = stim_mean_space[loc]
        mirad = get_deg_to_rad(mi, True)
        uprad = get_deg_to_rad(prior_mode, True)

        # set k ratio
        k_ratio = k_llh / k_prior
        if k_llh == k_prior == np.inf:
            k_ratio = 1
        else:
            raise Exception("Check k_prior or k_llh")

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

        if k_llh == np.inf or k_prior == np.inf:
            kpo = np.sqrt(
                k_llh ** 2
                + k_prior ** 2
                + 2 * k_prior * k_llh * cos(uprad - mirad)
            )
            raise Exception(
                """We have not yet solved Bayesian
                 integration when k_llh or k_prior is 
                 +inf"""
            )

        # create those posterior
        posterior[:, loc] = VonMises(p=True).get(
            stim_mean_space, upo, [kpo],
        )
    return posterior
