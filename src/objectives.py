import numpy as np

# test function
class TestObjective(object):
    def __init__(self):
        pass

    def objective(self, val, param):
        return sum(val**2) + sum(param**2)


class BayesianObjective(object):
    def __init__(self, val_target, param_prior, sigma_obs, sigma_prior):
        self.val_target = val_target
        self.param_prior = param_prior
        self.sigma_obs = sigma_obs
        self.sigma_prior = sigma_prior
        C = np.eye(self.val_target.size)
        self.set_obs_covariance(C*sigma_obs**2)
        self.set_prior_covariance(C*sigma_prior**2)

    def set_obs_covariance(self, C):
        self.cov_obs = C.copy()
        self.cov_obs_inv = np.linalg.inv(self.cov_obs)

    def set_prior_covariance(self, C):
        self.cov_prior = C.copy()
        self.cov_prior_inv = np.linalg.inv(self.cov_prior)
       
    def objective(self, val, param):
        assert(np.size(val) == np.size(self.val_target))
        assert(np.size(param) == np.size(self.param_prior))
        val_vector = val - self.val_target
        param_vector = param - self.param_prior
        val_vector = val_vector[1:-1]
        param_vector = param_vector
        J_obs = val_vector.transpose().dot(self.cov_obs_inv).dot(val_vector)
        J_prior = param_vector.transpose().dot(self.cov_prior_inv).dot(param_vector)
        J = 0.5*(J_obs + J_prior)
        return J

    def objective_jac(self, val, param, i):
        assert(np.size(val) == np.size(self.val_target))
        assert(np.size(param) == np.size(self.param_prior))
        J_obs = val[i] - self.val_target[i]
        return J_obs
