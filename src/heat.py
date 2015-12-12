import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from objectives import TestObjective
from schemes import diff2

class HeatBase(object):
    def __init__(self, T_inf, ngrid = 31, verbose=False, userealbeta=False):
        self.T_inf = T_inf
        self.grid = 1
        self.x = self.get_grid(ngrid)
        self.T = np.ones_like(self.x)*self.T_inf
        self.T = self.T.astype(np.complex)
        self.n = self.T.size
        self.dt = 1.0e1
        self.maxiter = 100
        self.h = 0.5
        self.eps_0 = 5.0e-4
        self.tol = 1e-6
        self.noise = np.random.randn(np.size(self.T))*0.1
        self.objective = TestObjective()
        self.verbose = verbose
        self.beta = np.ones_like(self.T)
        self.userealbeta = userealbeta
        
    def get_eps(self, T):
        eps = 1.0 + 5.0*np.sin(3.0*np.pi/200.0*T) + np.exp(0.02*T) + self.noise
        eps = eps*1e-4
        assert(eps.size == T.size)
        return eps

    def get_grid(self, n):
        x = np.loadtxt("y")
        xr = x[::-1]
        xx = np.zeros(2*x.size - 1)
        xx[:x.size] = x[:]
        xx[x.size:] = 1.0 - xr[1:]
        if n != 401:
            N = n
            xx = np.linspace(0.0, 1.0, N)
        return xx

    def calc_residual(self, T):
        x = self.x
        T_inf = self.T_inf
        eps = self.get_eps(T)
        h = self.h

        R = np.zeros_like(T)
        R = -diff2(x, T) - eps*(T_inf**4 - T**4) - h*(T_inf - T)
        R[0] = T[0]
        R[-1] = T[-1]
        return R

    def calc_residual_jacobian(self, T, dT=1e-25):
        n = np.size(T)
        dRdT = np.zeros([n, n], dtype=T.dtype)
        for i in range(n):
            T[i] = T[i] + 1j*dT
            R = self.calc_residual(T)
            dRdT[:,i] = np.imag(R[:])/dT
            T[i] = T[i] - 1j*dT
        return dRdT

    def calc_dt(self):
        return self.dt*np.ones(self.n)

    def step(self, T, dt):
        R = self.calc_residual(T)
        dRdT = self.calc_residual_jacobian(T)
        dt = self.calc_dt()
        A = np.zeros_like(dRdT)
        n = self.n
        for i in range(0, n):
            A[i,i] = 1./dt[i]
        A = A - dRdT
        dT = linalg.solve(A, R)
        l2norm = np.sqrt(sum(R**2))/np.size(R)
        return dT, l2norm


    def boundary(self, T):
        plt.ion()
        plt.figure(1)
        plt.clf()
        plt.plot(self.x, T)
        plt.pause(0.0001)

    def save(self, T):
        if self.verbose:
            np.savetxt("solution/T", T.astype(np.float64))
            np.savetxt("solution/x", self.x.astype(np.float64))

    def solve(self):
        T = np.copy(self.T)
        dt = self.dt
        for i in range(self.maxiter):
            dT, l2norm = self.step(T, dt)
            T[:] = T[:] + dT[:]
            #self.boundary(T)
            if self.verbose:
                print "Iteration: %i Norm: %1.2e"%(i, l2norm)
            self.save(T)
            if l2norm < self.tol:
                break
        if l2norm > self.tol:
            print "DID NOT CONVERGE"
        self.T[:] = T[:]
        if l2norm > self.tol:
            return False
        else:
            return True

    def calc_delJ_delbeta(self, T, beta):
        n = np.size(beta)
        dbeta = 1e-20
        dJdbeta = np.zeros_like(beta)
        for i in range(n):
            beta[i] = beta[i] + 1j*dbeta
            F = self.objective.objective(T, beta)
            dJdbeta[i] = np.imag(F)/dbeta
            beta[i] = beta[i] - 1j*dbeta
        return dJdbeta

    def calc_delJ_delT(self, T, beta):
        n = np.size(T)
        dT = 1e-30
        dJdT = np.zeros_like(T)
        for i in range(n):
            T[i] = T[i] + 1j*dT
            F = self.objective.objective(T, beta)
            dJdT[i] = np.imag(F)/dT
            T[i] = T[i] - 1j*dT
        return dJdT

    def calc_delJ_delT_jac(self, T, beta, i):
        n = np.size(T)
        dT = 1e-30
        dJdT = np.zeros_like(T)
        T[i] = T[i] + 1j*dT
        F = self.objective.objective_jac(T, beta, i)
        dJdT[i] = np.imag(F)/dT
        T[i] = T[i] - 1j*dT
        return dJdT


    def calc_psi(self, T, beta):
        dRdT = self.calc_residual_jacobian(T)
        dJdT = self.calc_delJ_delT(T, beta)
        psi = linalg.solve(dRdT.transpose(), -dJdT.transpose())
        return psi

    def calc_psi_jac(self, T, beta, i):
        dRdT = self.calc_residual_jacobian(T)
        dJdT = self.calc_delJ_delT_jac(T, beta, i)
        psi = linalg.solve(dRdT.transpose(), -dJdT.transpose())
        return psi

    def calc_delR_delbeta(self, T):
        nb = np.size(self.beta)
        n = np.size(T)
        dbeta = 1e-30
        dRdbeta = np.zeros([n,nb], dtype=T.dtype)
        for i in range(nb):
            self.beta[i] = self.beta[i] + 1j*dbeta
            R = self.calc_residual(T)
            dRdbeta[:,i] = np.imag(R[:])/dbeta
            self.beta[i] = self.beta[i] - 1j*dbeta
        return dRdbeta

    def calc_sensitivity(self):
        T = self.T
        beta = self.beta
        psi = self.calc_psi(T, beta)
        delJdelbeta = self.calc_delJ_delbeta(T, beta)
        delRdelbeta = self.calc_delR_delbeta(T)
        dJdbeta = delJdelbeta + psi.transpose().dot(delRdelbeta)
        return dJdbeta

    def calc_sensitivity_jac(self, i):
        T = self.T
        beta = self.beta
        psi = self.calc_psi_jac(T, beta, i)
        delRdelbeta = self.calc_delR_delbeta(T)
        dJdbeta = psi.transpose().dot(delRdelbeta)
        return dJdbeta
        
    def calc_hessian_fd(self):
        n = np.size(self.T)
        H = np.zeros([n, n])
        dbeta = 1e-5
        beta = self.beta.copy()
        for i in range(n):
            self.beta[i] = beta[i] - dbeta
            self.solve()
            dJdbeta_jm = self.calc_sensitivity()
            self.beta[i] = beta[i] + dbeta
            self.solve()
            dJdbeta_jp = self.calc_sensitivity()
            H[i,:] = (dJdbeta_jp - dJdbeta_jm)/(2.0*dbeta)
        self.beta[:] = beta[:]
        return H

    
    def calc_post_cov(self):
        #nb = np.size(self.beta)
        #n = np.size(self.T)
        #jac = np.zeros([n, n])
        #for i in range(n):
        #    jac[i, :] = self.calc_sensitivity_jac(i)
        #H = jac.transpose().dot(jac)/self.objective.sigma_obs**2 + np.eye(n)/self.objective.sigma_prior**2
        H = self.calc_hessian_fd()
        Cov = np.linalg.inv(H + 1e-10*np.eye(self.n))
        return Cov

    def calc_beta(self, T):
        T_inf = self.T_inf
        eps = self.get_eps(T)
        beta = eps/self.eps_0 + self.h/self.eps_0*(T_inf - T)/((T_inf**4 - T**4) + 1e-16)
        return beta

class HeatModel(HeatBase):
    def calc_residual(self, T):
        x = self.x
        T_inf = self.T_inf
        if self.userealbeta:
            beta = self.calc_beta(T)
        else:
            beta = self.beta
        eps = beta*self.eps_0
        h = self.h
        R = np.zeros_like(T)
        R = -diff2(x, T) - eps*(T_inf**4 - T**4)
        R[0] = T[0]
        R[-1] = T[-1]
        return R

if __name__ == "__main__":
    for i in range(10, 55, 5):
        heat = HeatModel(i)
        heat.solve()
        plt.figure(2)
        x, T = heat.x, heat.T
        p = plt.plot(x, T, '--')
        
        heat = HeatModel(i, userealbeta=True)
        heat.solve()
        plt.figure(2)
        x, T = heat.x, heat.T
        plt.plot(x, T, '.', color=p[0].get_color())

        heat = HeatBase(i)
        heat.solve()
        plt.figure(2)
        x, T = heat.x, heat.T
        p = plt.plot(x, T, 'x-', color=p[0].get_color())
        
        
    plt.ioff()
    plt.show()
