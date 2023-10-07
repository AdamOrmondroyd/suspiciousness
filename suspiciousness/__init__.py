"""
Collection of functions useful for calculating tensions between datasets,
and plotting the results.

For uncorrelated datasets, the following functions are available:
sus.logR, sus.logS, sus.logI, sus.logp, sus.p, sus.d

For correlated datasets, please instead use:
sus.correlated.logR, sus.correlated.logS, sus.correlated.logI,
sus.correlated.logp, sus.correlated.p, sus.correlated.d

For plotting, please use:
sus.cornerplot, sus.sigma8plot

The functions use wrappers, so that they can be provided with either
anesthetic.NestedSamples.stats (which is required to for the calculation),
or anesthetic.NestedSamples (which is turned into stats),
or even just a string label + the additional chains argument.

a = ac.read_chains("chains/uniform/act/act_polychord_raw/act")
b = ac.read_chains("chains/uniform/bao/bao_polychord_raw/bao")
ab = ac.read_chains("chains/uniform/actbao/actbao_polychord_raw/actbao")

astats = a.stats()
bstats = b.stats()
abstats = ab.stats()

The three below are equivalent:

sus.logR(astats, bstats, abstats)
sus.logR(a, b, ab)
sus.logR("act", "bao", "actbao", chains="chains/uniform")
"""
from suspiciousness.core import logR, logI, logS, logp, p, d
from suspiciousness import correlated
from suspiciousness.plot import cornerplot, sigma8plot
from suspiciousness.utils import read_cobaya_chains, samples, stats
