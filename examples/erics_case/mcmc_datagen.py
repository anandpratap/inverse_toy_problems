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
    ngrid = 21
    T_inf = 50.0
    obs_cov_type = "scalar"
    nsamples = 1000
    # generate observational data
    T_samples = np.zeros([nsamples, ngrid])
    beta_samples = np.zeros([nsamples, ngrid])
    i = 0
    while i < nsamples:
        heat = HeatBase(T_inf, ngrid=ngrid)
        status = heat.solve()
        if status:
            T_samples[i, :] = heat.T[:]
            beta_samples[i, :] = heat.calc_beta(heat.T)
            i = i + 1
            print "\rGenerating observational data %i of %i"%(i+1,nsamples),
        else:
            heat.T[:] = np.ones_like(heat.T)*heat.T_inf
    print "\n"
    Cov = np.cov(T_samples.T)
    sigma_vector = np.sqrt(np.diag(Cov[1:-1,1:-1]))
    sigma_obs = np.sqrt(np.mean(sigma_vector))
    sigma_prior = 0.8

    data = T_base_mean = np.mean(T_samples, axis=0)
    beta_base_mean = np.mean(beta_samples, axis=0)
    
    beta_prior = np.ones_like(beta_base_mean)

    objective = BayesianObjective(T_base_mean, beta_prior, sigma_obs, sigma_prior)

    if obs_cov_type == "scalar":
        objective.cov_obs_inv = np.linalg.inv(np.eye(ngrid-2)*sigma_obs**2)
    elif obs_cov_type == "vector":
        Cov_ = np.eye(ngrid-2)
        for i in range(ngrid-2):
            Cov_[i,i] = sigma_vector[i]**2
        objective.cov_obs_inv = np.linalg.inv(Cov_)
    elif obs_cov_type == "matrix":
        objective.cov_obs_inv = np.linalg.inv(Cov[1:-1,1:-1])
    else:
        raise ValueError("Wrong argument for covariance matrix type.")

    heat = HeatModel(T_inf, ngrid=ngrid)
    heat.solve()
    xi, Ti = heat.x, heat.T.copy()
    heat.objective = objective
    inverse = InverseSolver(heat)
    inverse.maxiter = 30
    inverse.nsamples = 2
    heat = inverse.solve()
    xf, Tf = heat.x, heat.T
    
    beta_map = heat.beta.astype(np.float64)
    beta_prior = beta_prior.astype(np.float64)
    data = data.astype(np.float64)
    sigma_obsv = np.ones_like(beta_map)*sigma_obs
    sigma_priorv = np.ones_like(beta_map)*sigma_prior
    np.savetxt('mcmc_input.dat', np.c_[data, np.sqrt(np.diag(Cov)), beta_prior, sigma_priorv, beta_map])

