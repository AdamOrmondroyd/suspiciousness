"""
Core functions for suspiciousness, for correlated datasets.
"""
from suspiciousness.utils import stats
from scipy.stats import chi2


@stats
def logR(h1, h0, show=False):
    """
    log Bayes factor for correlated datasets.

    logR = logZ_H0 - logZ_H1

    Parameters
    ----------
    h1: alternative hypothesis
    h0: null hypothesis
    show: bool. If True, print the mean and std of logR.

    Returns
    -------
    logR: anesthetic.samples.WeightedLabelledSeries
    """

    logR = h0.logZ - h1.logZ
    logR.name = r"$\log{R}$"
    if show:
        print(f"logR = {logR.mean()} ± {logR.std()}")
    return logR


@stats
def logS(h1, h0, show=False):
    """
    log suspiciousness for correlated datasets.

    logS = <logL_H0> - <logL_H1>

    Parameters
    ----------
    h1: alternative hypothesis
    h0: null hypothesis
    show: bool. If True, print the mean and std of logS.

    Returns
    -------
    logS: anesthetic.samples.WeightedLabelledSeries
    """

    logS = h0.logL_P - h1.logL_P
    logS.name = r"$\log{S}$"
    if show:
        print(f"logS = {logS.mean()} ± {logS.std()}")
    return logS


@stats
def logI(h1, h0, show=False):
    """
    log information ratio for correlated datasets.

    logI = D_KL(H1) - D_KL(H0)

    Parameters
    ----------
    h1: alternative hypothesis
    h0: null hypothesis
    show: bool. If True, print the mean and std of logI.

    Returns
    -------
    logI: anesthetic.samples.WeightedLabelledSeries
    """

    logI = h1.D_KL - h0.D_KL
    logI.name = r"$\log{I}$"
    if show:
        print(f"logI = {logI.mean()} ± {logI.std()}")
    return logI


@stats
def d(h1, h0, show=False):
    """
    Difference in Bayesian dimensionality between H1 and H0.

    d = d_G(H0) - d_G(H1)

    Parameters
    ----------
    h1: alternative hypothesis
    h0: null hypothesis
    show: bool. If True, print the mean and std of logI.

    Returns
    -------
    d: anesthetic.samples.WeightedLabelledSeries
    """
    d = h1.d_G - h0.d_G
    d.name = r"$d$"
    if show:
        print(f"d = {d.mean()} ± {d.std()}")
    return d


@stats
def logp(h1, h0, show=False):
    """
    log p-value.

    logp = log(1 - chi2.cdf(d-2*logS, d))

    Parameters
    ----------
    h1: alternative hypothesis
    h0: null hypothesis
    show: bool. If True, print the mean and std of logp.

    Returns
    -------
    logp: anesthetic.samples.WeightedLabelledSeries
    """
    _d = d(h1, h0)
    logp = chi2.logsf(_d-2*logS(h1, h0), _d)
    logp.name = r"$\log{p}$"
    if show:
        print(f"logp = {logp.mean()} ± {logp.std()}")
    return logp


@stats
def p(h1, h0, show=False):
    """
    p-value.

    p = 1 - chi2.cdf(d-2*logS, d)

    Parameters
    ----------
    h1: alternative hypothesis
    h0: null hypothesis
    show: bool. If True, print the mean and std of p.

    Returns
    -------
    p: anesthetic.samples.WeightedLabelledSeries
    """
    _d = d(h1, h0)
    p = chi2.sf(_d-2*logS(h1, h0), _d)
    p.name = r"$p$"
    if show:
        print(f"p = {p.mean()} ± {p.std()}")
    return p
