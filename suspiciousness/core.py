"""
Core functions for suspiciousness.
"""
from suspiciousness.utils import stats
from scipy.stats import chi2


@stats
def logR(a, b, ab, show=False):
    """
    log Bayes factor for correlated datasets.

    logR = logZ_H0 - logZ_H1

    Parameters
    ----------
    a : dataset A
    b : dataset B
    ab : combined dataset AB
    show: bool. If True, print the mean and std of logR.

    Returns
    -------
    logR: anesthetic.samples.WeightedLabelledSeries
    """

    logR = ab.logZ - a.logZ - b.logZ
    logR.name = r"$\log{R}$"
    if show:
        print(f"logR = {logR.mean()} ± {logR.std()}")
    return logR


@stats
def logS(a, b, ab, show=False):
    """
    log suspiciousness for correlated datasets.

    logS = <logL_H0> - <logL_H1>

    Parameters
    ----------
    a : dataset A
    b : dataset B
    ab : combined dataset AB
    show: bool. If True, print the mean and std of logS.

    Returns
    -------
    logS: anesthetic.samples.WeightedLabelledSeries
    """
    logS = ab.logL_P - a.logL_P - b.logL_P
    logS.name = r"$\log{S}$"
    if show:
        print(f"logS = {logS.mean()} ± {logS.std()}")
    return logS


@stats
def logI(a, b, ab, show=False):
    """
    log information ratio for correlated datasets.

    logI = D_KL(H1) - D_KL(H0)

    Parameters
    ----------
    a : dataset A
    b : dataset B
    ab : combined dataset AB
    show: bool. If True, print the mean and std of logI.

    Returns
    -------
    logI: anesthetic.samples.WeightedLabelledSeries
    """
    logI = a.D_KL + b.D_KL - ab.D_KL
    if show:
        print(f"logI = {logI.mean()} ± {logI.std()}")
    return logI


@stats
def d(a, b, ab, show=False):
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
    d = a.d_G + b.d_G - ab.d_G
    d.name = r"$d$"
    if show:
        print(f"d = {d.mean()} ± {d.std()}")
    return d


@stats
def logp(a, b, ab, show=False):
    """
    log p-value.

    logp = log(1 - chi2.cdf(d-2*logS, d))

    Parameters
    ----------
    a : dataset A
    b : dataset B
    ab : combined dataset AB
    show: bool. If True, print the mean and std of logp.

    Returns
    -------
    logp: anesthetic.samples.WeightedLabelledSeries
    """
    _d = d(a, b, ab)
    logp = chi2.logsf(_d-2*logS(a, b, ab), _d)
    logp.name = r"$\log{p}$"
    if show:
        print(f"logp = {logp.mean()} ± {logp.std()}")
    return logp


@stats
def p(a, b, ab, show=False):
    """
    p-value.

    p = 1 - chi2.cdf(d-2*logS, d)

    Parameters
    ----------
    a : dataset A
    b : dataset B
    ab : combined dataset AB
    show: bool. If True, print the mean and std of p.

    Returns
    -------
    p: anesthetic.samples.WeightedLabelledSeries
    """
    _d = d(a, b, ab)
    p = chi2.sf(_d-2*logS(a, b, ab), _d)
    p.name = r"$p$"
    if show:
        print(f"p = {p.mean()} ± {p.std()}")
    return p
