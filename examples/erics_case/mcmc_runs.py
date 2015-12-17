import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as mc

import theano
import theano.tensor as T 
from theano.compile.ops import as_op
sys.path.insert(1, "../../src")
from heat import HeatBase, HeatModel
from inverse import InverseSolver
from objectives import TestObjective, BayesianObjective
from mcmc import MCMCSampler   


if __name__ == "__main__":
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("--run", action="store_true", help="Run.")
    args = parser.parse_args()
    # class SimpleObjective(object):
    #     def __init__(self):
    #         pass
    #     def objective_jac(self, val, param, i):
    #         J_obs = val[i]
    #         return J_obs
            

    @as_op(itypes=[theano.tensor.dvector], otypes=[theano.tensor.dvector])
    def function(beta):
        ngrid = 21
        T_inf = 50.0
        heat = HeatModel(T_inf, ngrid=ngrid)
        heat.beta[1:-1] = beta.astype(np.complex)
        status = heat.solve()
        if status:
            tmp = heat.T.astype(np.float64)
        else:
            tmp = np.ones(ngrid)*(1e10)
        return tmp[1:-1]

    # @as_op(itypes=[theano.tensor.dvector, theano.tensor.dvector], otypes=[theano.tensor.dvector])
    # def function_grad(beta, z):
    #     ngrid = 21
    #     T_inf = 50.0
    #     heat = HeatModel(T_inf, ngrid=ngrid)
    #     heat.beta[1:-1] = beta.astype(np.complex)
    #     heat.objective = SimpleObjective()
    #     status = heat.solve()
    #     if status:
    #         tmp = heat.T.astype(np.float64)
    #     else:
    #         tmp = np.ones(ngrid)*(1e10)
    #     n = ngrid
    #     jac = np.zeros([n-2, n-2])
    #     for i in range(0,n-2):
    #         jac[i, :] = heat.calc_sensitivity_jac(i+1)[1:-1]
    #     tmp_1 = np.dot(jac.astype(np.float64).transpose(), z)
    #     return tmp_1

    # function_grad.grad = lambda *x: x[0]#*0

    # class Func(theano.Op):
    #     def __init__(self):
    #         pass

    #     def make_node(self, beta):
    #         # check that the theano version has support for __props__.
    #         assert hasattr(self, '_props'), "Your version of theano is too old to support __props__."
    #         beta = theano.tensor.as_tensor_variable(beta)
    #         assert beta.ndim == 1
    #         return theano.Apply(self, [beta], [beta.type()])
            
    #     def perform(self, node, inputs, output_storage):
    #         beta = inputs[0]
    #         z = output_storage[0]
    #         z[0] = function(beta).copy()

    #     def infer_shape(self, node, i0_shapes):
    #         return i0_shapes

    #     def grad(self, inputs, output_grads):
    #         beta = inputs[0]
    #         grad = function_grad(beta, output_grads[0])
    #         return [grad]


    d = np.loadtxt("mcmc_input.dat")
    Cov = np.loadtxt("mcmc_input_cov.dat")[1:-1,1:-1]
    tau = np.linalg.inv(Cov)
    data = d[1:-1,0]
    tau_obs = tau
    beta_prior = d[1:-1,2]
    tau_prior = np.eye(np.size(data))/0.8**2
    beta_map = d[1:-1,4]
    
    sampler = MCMCSampler(function, data, tau_obs, beta_prior, tau_prior, beta_map, save="trace_cov.sqlite", is_cov=True)
    with sampler.model:
        try:
            trace = mc.backends.sqlite.load("trace_cov.sqlite")
            start = trace[-1]
        except:
            start = None
            
    if args.run:
        trace = sampler.sample(100000, start=start)
    else:
        pass
    mc.traceplot(trace)
    mc.summary(trace)
    
    beta_ = trace['beta'][1000:,:]
    beta_mean = np.mean(beta_, axis=0)
    beta_std = np.std(beta_, axis=0)
    plt.figure()
    plt.plot(beta_mean)
    plt.plot(trace["beta"][-1], '.')
    plt.plot(beta_map)
    #plt.figure()
    #plt.plot(function(beta_map[:]))
    #plt.plot(data, 'x')
    plt.figure()
    plt.plot(beta_std)
    plt.show()
