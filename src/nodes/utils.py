from collections import defaultdict

import numpy as np
import pandas as pd
from numpy import arctan2, cos, sin
from scipy.optimize import fmin
from src.nodes.data import VonMises
from src.nodes.util import (
    get_deg_to_rad,
    get_rad_to_deg,
    is_empty,
)


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
            "vonMisesPrior"
        prior_mode: (float): mode of the prior

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

    # set model architecture
    model = (readout,)

    task_params = (
        (
            data["stim_std"],
            prior_mode,
            data["prior_std"],
            prior_shape,
        ),
    )

    # set data
    true = data["stim_mean"]
    pred = data["estimate"]

    # fit
    neglogl = fmin(
        get_logl,
        model_params,
        args=(true, pred, task_params, model),
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
    model: tuple,
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
                readout=model[0],
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

    # set stimulus feature space
    # e.g. motion direction
    stim_mean_space = np.arange(0, 360, 1)

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
        for i in range(len(percept)):
            map_loc = posterior[:, i] == np.max(
                posterior[:, i]
            )
            n_maps = sum(map_loc)
            percept[i, 0:n_maps] = stim_mean_space[map_loc]
    else:
        raise NotImplementedError(
            """readout has not yet 
            been implemented"""
        )

    percept[percept == 0] = 360

    # run sanity check
    if (percept > 360).any() or (percept < 1).any():
        raise ValueError("percept must be >=1 and <=360")

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
    n_percept_each_mi = sum(~np.isnan(percept), 0)

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

    from ipdb import set_trace

    set_trace()
    # sort by percept
    the_matx[the_matx[:, 1].argsort()]

    # re-extract
    ul = the_matx[:, 0]
    percept = the_matx[:, 1]
    PperceptgivenMi = the_matx[:, 2 : 2 + len(stim_mean)]
    PmiGivenDi = the_matx[:, 2 + len(stim_mean) :]

    # likelihood of each percept estimate
    PperceptgivenDi = PperceptgivenMi * PmiGivenDi

    # Set the likelihood of percepts not produced at 0 because the model cannot
    # produce those estimates even at a reasonable resolutions of motion
    # direction
    from ipdb import set_trace

    set_trace()

    return None


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
