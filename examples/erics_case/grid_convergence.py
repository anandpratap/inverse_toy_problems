import sys
import argparse
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
sys.path.insert(1, "../../src")
from heat import HeatBase, HeatModel
from objectives import TestObjective, BayesianObjective
from inverse import InverseSolver
if __name__ == "__main__":
    cov_choices = ['scalar', 'vector', 'matrix']
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("--nsamples", type=int, default=100, required=True, help="Number of samples.")
    parser.add_argument("--obs_cov_type", nargs=1, choices=cov_choices, required=True, help="Type of observational covariance matrix.")
    parser.add_argument("--save_figures", action="store_true", help="Force boundary.")

    args = parser.parse_args()
    ngrids = [11, 51, 101, 501, 1001]#, 5001]
    ng = len(ngrids)
    nmax = ngrids[-1]
    xmax = np.linspace(0.0, 1.0, nmax)
    T_map_grids = np.zeros([ng, nmax])
    beta_map_grids = np.zeros([ng, nmax])
    T_std_grids = np.zeros([ng, nmax])
    beta_std_grids = np.zeros([ng, nmax])
    for idx, ngrid in enumerate(ngrids):
        nsamples = args.nsamples
        T_inf = 50.0
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
        f = interp1d(xb, T_post_mean)
        T_map_grids[idx,:] = f(xmax)

        f = interp1d(xb, beta_post_mean)
        beta_map_grids[idx,:] = f(xmax)
        
        f = interp1d(xb, T_post_std)
        T_std_grids[idx,:] = f(xmax)

        f = interp1d(xb, beta_post_std)
        beta_std_grids[idx,:] = f(xmax)

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
        np.savetxt("data/grid_convergence_%i_%s.dat"%(ngrid,obs_cov_type), np.c_[xi,T_base_mean,T_prior_mean,T_post_mean,T_prior_std,T_post_std,beta_prior_mean,beta_post_mean,beta_prior_std,beta_post_std])


plt.figure()
error_T_mean = np.linalg.norm(T_map_grids[:,:] - T_map_grids[-1,:], axis=1)
plt.loglog(ngrids, error_T_mean, "rx-")
plt.xlabel("N")
plt.ylabel("Error T Post mean")
plt.savefig("figures/grid_convergence_%s_error_T_mean.pdf"%obs_cov_type)
plt.figure()
error_T_std = np.linalg.norm(T_std_grids[:,:] - T_std_grids[-1,:], axis=1)
plt.loglog(ngrids, error_T_std, "rx-")
plt.xlabel("N")
plt.ylabel("Error T Post Standard Deviation")
plt.savefig("figures/grid_convergence_%s_error_T_std.pdf"%obs_cov_type)

plt.figure()
error_beta_mean = np.linalg.norm(beta_map_grids[:,:] - beta_map_grids[-1,:], axis=1)
plt.loglog(ngrids, error_beta_mean, "rx-")
plt.xlabel("N")
plt.ylabel(r"Error $\beta$ Post mean")
plt.savefig("figures/grid_convergence_%s_error_beta_mean.pdf"%obs_cov_type)

plt.figure()
error_beta_std = np.linalg.norm(beta_std_grids[:,:] - beta_std_grids[-1,:], axis=1)
plt.loglog(ngrids, error_beta_std, "rx-")
plt.xlabel("N")
plt.ylabel(r"Error $\beta$ Post std")
plt.savefig("figures/grid_convergence_%s_error_beta_std.pdf"%obs_cov_type)
plt.show()
