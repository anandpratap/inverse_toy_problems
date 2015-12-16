import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as mc
import theano.tensor as T 
from theano.compile.ops import as_op
sys.path.insert(1, "../../src")
from heat import HeatBase, HeatModel
from inverse import InverseSolver
from objectives import TestObjective, BayesianObjective
from mcmc import MCMCSampler   


if __name__ == "__main__":
    @as_op(itypes=[T.dvector], otypes=[T.dvector])
    def function(beta):
        ngrid = 21
        T_inf = 50.0
        heat = HeatModel(T_inf, ngrid=ngrid)
        heat.beta = beta.astype(np.complex)
        status = heat.solve()
        if status:
            tmp = heat.T.astype(np.float64)
        else:
            tmp = np.ones_like(heat.T.astype(np.float64))*(1e10)
        return tmp

    #function.grad = lambda *x: x[0]
    d = np.loadtxt("mcmc_input.dat")
    data = d[:,0]
    sigma_obsv = d[:,1]
    beta_prior = d[:,2]
    sigma_priorv = d[:,3]
    beta_map = d[:,4]
    sampler = MCMCSampler(function, data, sigma_obsv, beta_prior, sigma_priorv, beta_map, save='trace.sqlite')
    trace = sampler.sample(200000)
    mc.traceplot(trace)
    mc.summary(trace)
    plt.show()
