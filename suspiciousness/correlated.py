from suspiciousness.utils import stats
from scipy.stats import chi2


@stats
def logR(h1, h0, show=False):
    logR = h0.logZ - h1.logZ
    if show:
        print(f"logR = {logR.mean()} ± {logR.std()}")
    return logR


@stats
def logS(h1, h0, show=False):
    logS = h0.logL_P - h1.logL_P
    if show:
        print(f"logS = {logS.mean()} ± {logS.std()}")
    return logS


@stats
def logI(h1, h0, show=False):
    logI = h0.D_KL - h1.D_KL
    if show:
        print(f"logI = {logI.mean()} ± {logI.std()}")
    return logI


@stats
def bayesian_d(h1, h0):
    return h1.d_G - h0.d_G


@stats
def logp(h1, h0, show=False):
    d = bayesian_d(h1, h0)
    logp = chi2.logsf(d-2*logS(h1, h0), d)
    if show:
        print(f"logp = {logp.mean()} ± {logp.std()}")
    return logp


@stats
def p(h1, h0, show=False):
    d = bayesian_d(h1, h0)
    p = chi2.sf(d-2*logS(h1, h0), d)
    if show:
        print(f"p = {p.mean()} ± {p.std()}")
    return p
