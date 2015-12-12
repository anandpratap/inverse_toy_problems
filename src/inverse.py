import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from heat import HeatBase, HeatModel
from objectives import TestObjective, BayesianObjective

class InverseSolver(object):
    def __init__(self, eqn):
        self.eqn = eqn
        self.maxiter = 40
        self.dostats = False
        self.algo = "bfgs"
        self.stepsize = 0.1

    def sample_prior(self, nsamples):
        beta_samples_prior = np.zeros([nsamples, self.eqn.beta.size])
        T_samples_prior = np.zeros([nsamples, self.eqn.beta.size])
        cov_prior = self.eqn.objective.cov_prior
        R = np.linalg.cholesky(cov_prior)
        i = 0
        while i < nsamples:
            s = np.random.randn(self.eqn.beta.size)
            beta_samples_prior[i,:] = 1.0 + R.dot(s)
            self.eqn.beta[:] = beta_samples_prior[i,:]
            status = self.eqn.solve()
            if status:
                T_samples_prior[i,:] = self.eqn.T[:]
                print "\rPrior samples %i of %i"%(i+1, nsamples),
                i = i + 1
            else:
                heat.T[:] = np.ones_like(heat.T)*heat.T_inf
        print "\n"
        self.eqn.beta[:] = 1.0
        return beta_samples_prior, T_samples_prior


    def sample_post(self, nsamples):
        beta_samples_post = np.zeros([nsamples, self.eqn.beta.size])
        T_samples_post = np.zeros([nsamples, self.eqn.beta.size])
        cov_post = self.eqn.calc_post_cov()
        R = np.linalg.cholesky(cov_post + 1e-5*np.eye(self.eqn.n))
        beta_map = self.eqn.beta.copy()
        i = 0
        while i < nsamples:
            s = np.random.randn(self.eqn.beta.size)
            beta_samples_post[i,:] = beta_map[:] + R.dot(s)
            self.eqn.beta[:] = beta_samples_post[i,:]
            status = self.eqn.solve()
            if status:
                T_samples_post[i,:] = self.eqn.T[:]
                print "\rPosterior samples %i of %i"%(i+1, nsamples),
                i = i + 1
            else:
                heat.T[:] = np.ones_like(heat.T)*heat.T_inf
        print "\n"
        self.eqn.beta[:] = beta_map[:]
        return beta_samples_post, T_samples_post

    def get_stepsize(self):
        return 1.0

    def step_sd(self):
        self.eqn.solve()
        dJdbeta = self.eqn.calc_sensitivity()
        dJdbeta_norm= dJdbeta/np.linalg.norm(dJdbeta)
        pk = -dJdbeta_norm
        stepsize = self.linesearch(self.stepsize, pk)
        self.eqn.beta += stepsize*pk
    
    def step_bfgs(self):
        self.eqn.solve()
        dJdbeta = self.eqn.calc_sensitivity()
        dJdbeta_norm= dJdbeta/np.linalg.norm(dJdbeta)
        
        if self.i == 0:
            self.B = np.eye(np.size(self.eqn.beta))
        else:
            yk = (dJdbeta_norm - self.dJdbeta_norm)[np.newaxis].T
            sk = self.sk[np.newaxis].T
            term_1_num = yk.dot(yk.transpose())
            term_1_den = yk.transpose().dot(sk)
            term_2_num = self.B.dot(sk.dot(sk.transpose().dot(self.B)))
            term_2_den = sk.transpose().dot(self.B.dot(sk))
            self.B = self.B + term_1_num/term_1_den - term_2_num/term_2_den

        pk = np.linalg.solve(self.B, -dJdbeta_norm)
        pk = pk/np.linalg.norm(pk)
        stepsize = self.linesearch(1.1, pk)
        sk = stepsize*pk
        self.eqn.beta += sk
        self.sk = sk
        self.dJdbeta_norm = dJdbeta_norm

    def linesearch(self, stepsize, pk):
        beta_ = self.eqn.beta.copy()
        q_ = self.eqn.T.copy()
        J_ = self.eqn.objective.objective(self.eqn.T, self.eqn.beta)
        for i in range(20):
            self.eqn.T[:] = q_[:]
            self.eqn.beta = beta_ + pk*stepsize
            self.eqn.solve()
            J = self.eqn.objective.objective(self.eqn.T, self.eqn.beta)
            if J < J_:
                self.eqn.beta[:] = beta_[:]
                break
            else:
                stepsize /= 2.0
        self.eqn.beta[:] = beta_[:]
        return stepsize


    def calculate_hessian(self):
        pass

    def calculate_cholesky(self):
        H = self.calculate_hessian()
        Cov = np.linalg.inv(H)
        R = np.chol(Cov)
        return R
        
    def solve(self):
        self.J_base = 1.0
        self.beta_samples_prior, self.T_samples_prior = self.sample_prior(self.nsamples)
        for i in range(self.maxiter):
            self.i = i
            if self.algo == "sd":
                self.step_sd()
            else:
                self.step_bfgs()
            J = self.eqn.objective.objective(self.eqn.T, self.eqn.beta)
            if i == 0:
                self.J_base = J
            print 30*"#", "ITER: ", i, "J: ", J, "J_rel:", J/self.J_base
            np.savetxt("inverse_solution/beta.%i"%i, np.float64(self.eqn.beta))
        self.beta_samples_post, self.T_samples_post = self.sample_post(self.nsamples)
        return self.eqn
            
        
if __name__ == "__main__":
    cov_choices = ['scalar', 'vector', 'matrix']
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("--nsamples", type=int, default=100, required=True, help="Number of samples.")
    parser.add_argument("--ngrid", type=int, default=21, required=True, help="Number of grid points, use negative value to read from file.")
    parser.add_argument("--T_inf", type=float, default=50.0, required=True, help="T infinity")
    parser.add_argument("--obs_cov_type", nargs=1, choices=cov_choices, required=True, help="Type of observational covariance matrix.")
    parser.add_argument("--save_figures", action="store_true", help="Force boundary.")

    args = parser.parse_args()

    nsamples = args.nsamples
    ngrid = args.ngrid
    T_inf = args.T_inf
    obs_cov_type = args.obs_cov_type[0]
    if_save_figures = args.save_figures

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

    T_base_mean = np.mean(T_samples, axis=0)
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
    inverse.nsamples = nsamples
    heat = inverse.solve()
    xf, Tf = heat.x, heat.T
    
    beta_prior_mean = np.mean(inverse.beta_samples_prior, axis=0)
    T_prior_mean = np.mean(inverse.T_samples_prior, axis=0)
    beta_prior_std = np.std(inverse.beta_samples_prior, axis=0)
    T_prior_std = np.std(inverse.T_samples_prior, axis=0)

    beta_post_mean = np.mean(inverse.beta_samples_post, axis=0)
    T_post_mean = np.mean(inverse.T_samples_post, axis=0)
    beta_post_std = np.std(inverse.beta_samples_post, axis=0)
    T_post_std = np.std(inverse.T_samples_post, axis=0)

    xb = xi
    plt.figure()
    plt.plot(xb[1:-1], T_base_mean[1:-1], 'b.', label="Base")
    plt.plot(xi[1:-1], T_prior_mean[1:-1], 'g-', label="Prior")
    plt.fill_between(xi[1:-1], T_prior_mean[1:-1] - 2*T_prior_std[1:-1], T_prior_mean[1:-1] + 2*T_prior_std[1:-1], color='g', alpha=0.2)
    plt.plot(xf[1:-1], T_post_mean[1:-1], 'r-', label="Posterior")
    plt.fill_between(xi[1:-1], T_post_mean[1:-1] - 2*T_post_std[1:-1], T_post_mean[1:-1] + 2*T_post_std[1:-1], color='r', alpha=0.2)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$T$")
    if if_save_figures:
        fig_name = "figures/%s_%i_%i_%i_%s.pdf"%(obs_cov_type, T_inf, ngrid, nsamples, "T")
        plt.savefig(fig_name)

    plt.figure()
    plt.plot(xb[1:-1], beta_base_mean[1:-1], 'b.', label="Base")
    plt.plot(xi[1:-1], np.ones_like(xi[1:-1]), 'g-', label="Prior")
    plt.plot(xi[1:-1], beta_post_mean[1:-1], 'r-', label="Posterior")
    plt.fill_between(xi[1:-1], beta_post_mean[1:-1] - 2*beta_post_std[1:-1], beta_post_mean[1:-1] + 2*beta_post_std[1:-1], color='r', alpha=0.2)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$\beta(x)$")
    if if_save_figures:
        fig_name = "figures/%s_%i_%i_%i_%s.pdf"%(obs_cov_type, T_inf, ngrid, nsamples, "beta")
        plt.savefig(fig_name)


    plt.figure()
    plt.semilogy(xi[1:-1], np.std(beta_samples, axis=0)[1:-1], "b.", label="Base")
    plt.semilogy(xi[1:-1], beta_prior_std[1:-1], "g", label="Prior")
    plt.semilogy(xi[1:-1], beta_post_std[1:-1], "r", label="Posterior")
    plt.xlabel(r"$x$")
    plt.ylabel(r"Standard Deviation ($\sigma$)")
    if if_save_figures:
        fig_name = "figures/%s_%i_%i_%i_%s.pdf"%(obs_cov_type, T_inf, ngrid, nsamples, "std")
        plt.savefig(fig_name)

    plt.show()


    

