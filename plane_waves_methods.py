import numpy as np
import sympy
import math

from scipy import integrate

class Plane_waves_methods:

    def __init__(self, tools):
        self.tools = tools

    # Low depth quantum simulation of materials (babbush2018low) Trotter
    def low_depth_trotter(self, N, eta, Omega, eps_PEA, eps_HS, eps_S):
        
        #TODO eta = number of electrons
        t = 4.7/eps_PEA
        sum_1_nu = 4*np.pi(np.sqrt(3)*N**(1/3)/2 - 1) + 3 - 3/N**(1/3) + 3*self.tools.I(N**(1/3))
        max_V = eta**2/(2*np.pi*Omega**(1/3))*sum_1_nu
        max_U = eta**2/(np.pi*Omega**(1/3))*sum_1_nu
        nu_max = 3*(N**(1/3))**2
        max_T = 2*np.pi**2*eta/(Omega**(2/3))* nu_max
        
        r = np.sqrt(2*t**3/eps_HS *(max_T**2*(max_U + max_V) + max_T*(max_U + max_V)**2))

        eps_SS = eps_S/(2*N +N*(N-1) + N*np.log(N/2) + 8*N)
        
        exp_UV_cost = (4*N**2 + 4*N)*np.log(1/eps_SS)
        FFFT_cost = (2 + 4*np.log(1/eps_SS))*N*np.log(N) + 4*N*np.log(1/eps_SS)
        exp_T_cost = 32*N*np.log(1/eps_SS)
        
        return r*(exp_UV_cost + FFFT_cost + exp_T_cost)

    # Low depth quantum simulation of materials (babbush2018low) Taylor
    def low_depth_taylor(self, N, lambd, Lambd, eps_PEA, eps_HS, eps_S, Ham_norm):
        '''To be used in plane wave basis'''
        t = 4.7/eps_PEA
        r = t*Lambd/np.log(2)
        
        K_list = []
        
        for m_j in range(0, int(np.ceil(np.log(r)))):
            
            t_j = 2**m_j
            epsilon_HS_mj = eps_HS / r * 2**m_j
        
            K = np.ceil(np.log2(t_j/epsilon_HS_mj) / np.log2( np.log2 (t_j/epsilon_HS_mj)))
            K_list.append(K)
            
        epsilon_SS = eps_S /np.sum([3*2*(2*K) for K in K_list]) # The extra two is because Uniform requires 2 Rz gates
        
        mu = np.ceil(np.log(2*np.sqrt(2)*Lambd/eps_PEA) + np.log(1 + eps_PEA/(8*lambd)) + np.log(1 - (Ham_norm/lambd)**2))
        
        result = 0
        
        for m_j in range(0, int(np.ceil(np.log(r)))):
            
            t_j = 2**m_j
            epsilon_HS_mj = eps_HS / r * 2**m_j
        
            K = np.ceil(np.log2(t_j/epsilon_HS_mj) / np.log2( np.log2 (t_j/epsilon_HS_mj)))
        
            prepare_beta = K*(6*N+40*np.log(N)+16*np.log(1/epsilon_SS) + 10*mu)
            select_V = K*(12*N+8*np.log(N))
            
            result += 3*(2*prepare_beta + select_V)*t_j
        
        return result

    # Low depth quantum simulation of materials (babbush2018low) On-the fly
    def low_depth_taylor_on_the_fly(self, N, eta, lambd, Omega, eps_PEA, eps_HS, eps_S, eps_tay, Ham_norm, J, x_max):
        '''To be used in plane wave basis
        J: Number of atoms
        '''
        Lambd = (2*eta+1)*N**3 / (2*Omega**(1/3)*np.pi)
        t = 4.7/eps_PEA
        r = t*Lambd/np.log(2)
        
        mu = np.ceil(np.log(2*np.sqrt(2)*Lambd/eps_PEA) + np.log(1 + eps_PEA/(8*lambd)) + np.log(1 - (Ham_norm/lambd)**2))
        #K_list = []
        
        #epsilon_SS = epsilon_S /np.sum([3*2*(2*K) for K in K_list])
            
        #TODO x_max = max value of one dimension
        x = sympy.Symbol('x')
        K =  np.ceil(np.log2(r/eps_HS) / np.log2( np.log2 (r/eps_HS)))
        number_sample = 2*K* 3* 2* int(np.ceil(np.log(r)))
        eps_tay_s = eps_tay / number_sample
        order = self.tools.order_find(function = math.cos(x), x0 = 1, e = eps_tay_s, xeval = x_max)
        sample_w = 70*np.log(N)**2 + 29* np.log(N) + (21+14)*order/2*np.log(N)**2 + 2*order*np.log(N) + J*(35*order/2 + 63 + 2*order/np.log(N))*np.log(N)**2
        kickback = 32*np.log(mu)

        result = 0
        
        for m_j in range(0, int(np.ceil(np.log(r)))):
            
            t_j = 2**m_j
            epsilon_HS_mj = eps_HS / r * 2**m_j
        
            #K_list.append(K)
        
            K = np.ceil(np.log2(t_j/epsilon_HS_mj) / np.log2( np.log2 (t_j/epsilon_HS_mj)))
        
            prepare_W = 2*sample_w + kickback
            prepare_beta = K*prepare_W
            select_H = (12*N + 8*np.log(N))
            select_V = K*select_H
            
            result += 3*(2*prepare_beta + select_V)*t_j
        
        return result

    def f(self, x, y):
            return 1/(x**2 + y**2)
    def I(self, N0):
        return integrate.nquad(self.tools.f, [[1, N0],[1, N0]])[0]