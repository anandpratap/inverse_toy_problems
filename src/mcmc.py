import numpy as np
import matplotlib.pyplot as plt
import pymc3 as mc
from scipy import optimize

class MCMCSampler(object):
    def __init__(self, function, data, tau_obs, beta_prior, tau_prior, beta_map, njobs=1, save=None, is_cov=False, method=None):
        self.function = function
        self.beta_map = beta_map
        self.tau_obs = tau_obs

        self.beta_prior = beta_prior
        self.tau_prior = tau_prior
        
        self.data = data
        
        self.model = mc.Model()
        self.n_beta = np.size(self.beta_map)
        self.n_data = np.size(self.data)

        self.start = {'beta':beta_map}
        self.njobs = njobs
        self.save = save
        self.is_cov = is_cov
        self.method = method
        # test run of the function
        #q = function(self.beta_map)
        #assert(np.size(q) == np.size(self.data))
        assert(self.n_beta == np.size(self.beta_prior))
        assert(self.n_beta == np.size(np.diag(self.tau_prior)))
        assert(self.n_data == np.size(np.diag(self.tau_obs)))
        self.setup()
                
    def setup(self):
        with self.model:
            # Priors for unknown model parameters
            if self.is_cov:
                self.beta = beta = mc.MvNormal('beta', mu=self.beta_prior, tau=self.tau_prior, shape=self.n_beta)
            else:
                sd_prior = np.sqrt(np.diag(np.linalg.inv(self.tau_prior)))
                self.beta = beta = mc.Normal('beta', mu=self.beta_prior, sd=sd_prior, shape=self.n_beta)
            
            # Expected value of outcome
            mu = self.function(beta)
            
            # Likelihood (sampling distribution) of observations
            if self.is_cov:
                Y_obs = mc.MvNormal('Y_obs', mu=mu, tau=self.tau_obs, observed=self.data, shape=self.n_data)
            else:
                sd_obs = np.sqrt(np.diag(np.linalg.inv(self.tau_obs)))
                Y_obs = mc.Normal('Y_obs', mu=mu, sd=sd_obs, observed=self.data, shape=self.n_data)


    def sample(self, nsamples, start=None):
        with self.model:
            if self.method is None:
                step = mc.Metropolis(vars=[self.beta])
            elif self.method == "NUTS":
                step = mc.NUTS(vars=[self.beta])
            else:
                raise ValueError("Invalid method!")
                
            if self.save is None:
                backend = mc.backends.NDArray()
            else:
                backend = mc.backends.SQLite(self.save)
            if start is None:
                trace = mc.sample(nsamples, step=step, start=self.start, njobs=self.njobs, trace=backend)
            else:
                trace = mc.sample(nsamples, step=step, start=start, njobs=self.njobs, trace=backend)
        return trace




if __name__ == "__main__":
    def real_func():
        x = np.linspace(0.01, 1.0, 10)
        f = x + np.random.randn(len(x))*0.01
        return f
        
    def model_func(beta):
        x = np.linspace(0.01, 1.0, 10)
        f = beta
        return f

    data = real_func()
    tau_obs = np.eye(10)/.01**2
    tau_prior = np.eye(10)/1.0**2
    beta_prior = np.ones_like(data)*1.0
    beta_map = np.linspace(0.01, 1.0, 10) + np.random.randn(10)*0.1
    sampler = MCMCSampler(model_func, data, tau_obs, beta_prior, tau_prior, beta_map, is_cov=False, method=None)
    trace = sampler.sample(2000)
    mc.summary(trace)
    mc.traceplot(trace)
    plt.figure()
    plt.plot(beta_map, label='ACTUAL')
    plt.plot(np.mean(trace['beta'][:,:], axis=0), label='MCMC')
    plt.show()
