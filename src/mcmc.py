import numpy as np
import matplotlib.pyplot as plt
import pymc3 as mc
from scipy import optimize

class MCMCSampler(object):
    def __init__(self, function, data, sigma_obs, beta_prior, sigma_prior, beta_map, njobs=1, save=None):
        self.function = function
        self.beta_map = beta_map
        self.sigma_obs = sigma_obs

        self.beta_prior = beta_prior
        self.sigma_prior = sigma_prior
        
        self.data = data
        
        self.model = mc.Model()
        self.n_beta = np.size(self.beta_map)
        self.n_data = np.size(self.data)

        self.start = {'beta':beta_map}
        self.njobs = njobs
        self.save = save

        # test run of the function
        #q = function(self.beta_map)
        #assert(np.size(q) == np.size(self.data))
        assert(self.n_beta == np.size(self.beta_prior))
        assert(self.n_beta == np.size(self.sigma_prior))
        assert(self.n_data == np.size(self.sigma_obs))
        self.setup()
                
    def setup(self):
        with self.model:
            # Priors for unknown model parameters
            self.beta = beta = mc.Normal('beta', mu=self.beta_prior, sd=self.sigma_prior, shape=self.n_beta)
            # Expected value of outcome
            mu = self.function(beta)
            # Likelihood (sampling distribution) of observations
            Y_obs = mc.Normal('Y_obs', mu=mu, sd=self.sigma_obs, observed=self.data)


    def sample(self, nsamples, start=None):
        with self.model:
            step = mc.Metropolis(vars=[self.beta])
            if self.save is None:
                backend = mc.backends.ndarray()
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
        f = x #+ np.random.randn(len(x))*0.01
        return f
        
    def model_func(beta):
        x = np.linspace(0.01, 1.0, 10)
        f = beta
        return f

    data = real_func()
    sigma_obs = np.ones_like(data)*0.01
    sigma_prior = np.ones_like(data)*1.0
    beta_prior = np.ones_like(data)*1.0
    beta_map = np.linspace(0.01, 1.0, 10)

    sampler = MCMCSampler(model_func, data, sigma_obs, beta_prior, sigma_prior, beta_map, save='trace.sqlite')
    trace = sampler.sample(10000)
    mc.summary(trace)
    mc.traceplot(trace)
    plt.show()
