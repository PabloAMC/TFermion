import numpy as np

class Trotter_based_methods:

    def __init__(self):
        pass

    # qDrift and Trotterization

    # The first algorithm we would like to estimate its complexity is the q-Drift protocol from appendix A. The cost is ,
    # $$3n(10+12\\log \\epsilon^{-1})\\log N  = 3\\frac{27\\pi^2}{2}\\frac{\\lambda^2}{\\delta_E^2 P_f}(10+12\\log \\epsilon^{-1})\\log N$$,
    # where $\\lambda = \\sum a_\\gamma $ for the Hamiltonian $H = a_\\gamma H_\\gamma$;,
    # $\\delta_E$ is the error in Phase Estimation (an arbitrary parameter chosen by the user);,
    # and $P_f = \\frac{3}{2}p_f$ the probability of failure (also chosen by the user). The $\\epsilon$ parameter is given by the smallest of,
    # $$\\epsilon_j = \\epsilon_{tot}\\frac{2^j}{2(2^m-1)};$$ ,
    # We also need that,
    # $$ n = 4\\frac{\\pi^2(2^m-1)^2}{\\epsilon_{tot}}$$ with,

    # $$m = q +\\log_2 \\left(\\frac{1}{2p_f} + \\frac{1}{2}\\right),$$,
    # $\\delta_E = 2\\lambda\\delta$,  $q = \\log_2 \\delta -1$; and $P_f = p_f +2\\epsilon_{tot}$

    def calc_qdrift_resources(self, lambd, N, epsilon_PEA, epsilon_HS, epsilon_S):

        deltaE = epsilon_PEA
        eps_tot = epsilon_HS

        P_failure = 6*eps_tot

        r = ((27*np.pi**2/2)*(lambd/deltaE)**2) / P_failure**3

        #delta = deltaE/(2*lambd)
        #q = np.log2(1/delta)-1

        #pf = 2/3*P_failure
        #sanity check
        #assert (pf +2*eps_tot)/P_failure == 1
        
        #m = q + np.log2(1/(2*pf)+1/2)
        
        # Another sanity check. This should coincide
        #eps_tot_2 = 4*(np.pi*2**m-1)**2/n
        
        # error in individual rotations
        #epsilon_SS = (eps_tot/(2*np.pi*(2**m-1)))**2/N # Same as below. The N comes from Jordan Wigner implementation
        epsilon_SS = epsilon_S/(r*N) # from eq 39 same as the one above

        rost_cost_factor = N*self.tools.c_rotation_synthesis(epsilon_SS)
        
        return rost_cost_factor*r

    # For the randomised Hamiltonian approach, the equations are similar. However, now $p_f = 3/4P_f$ and ,
    # $$n = 8\\Gamma^2\\left(\\frac{ \\pi^3 \\Lambda^3}{8\\delta_E^3}\\right)^{1/2}\\left(\\frac{1+p_f}{p_f}\\right)^{3/2}\\frac{1}{\\epsilon_{tot}^{1/2}} 
    # = 4.35\\sqrt{8}\\pi^{3/2}\\Gamma^2 \\frac{\\Lambda^{3/2}}{\\delta_E^{3/2}P_f^2}$$ 
    def calc_rand_ham_resources(self, Lambd, lambd, Gamma, N, epsilon_PEA, epsilon_HS, epsilon_S):

        deltaE = epsilon_PEA
        eps_tot = epsilon_HS

        P_failure = 8*eps_tot

        r = 4.35*np.sqrt(8)*(np.pi*Lambd/deltaE)**(3/2) *(Gamma/ P_failure)**2

        # error in individual rotations
        #Lambda_A = Lambd/(2*lambd)
        #delta = deltaE/(2*lambd)
        #q = np.log2(1/delta)-1
        
        #pf = 3/4*P_failure
        #eps_tot = P_failure/8
        #sanity check
        #assert (pf +2*eps_tot)/P_failure == 1
        
        #m = q + np.log2(1/(2*pf)+1/2)
        
        #n1 = 8*Gamma**2 * ( 2**(m+1)*np.pi**3*Lambda_A**3/eps_tot  )**(1/2) *2*(2**m-1)
        
        # Another sanity check. This should coincide
        #eps_tot_2 = ( 8*Gamma**2* (np.pi*Lambd/(2*deltaE))**(3/2)* ((1+pf)/pf)**(3/2) /n1)**2
        #eps_tot_3 = 1/ ( 4.35* (1/P_failure)**2  *  (pf/(1+pf))**(3/2)  )**2
        
        #n2 = 8*Gamma**2 * ( 2**(m+1)*np.pi**3*Lambda_A**3/eps_tot_2  )**(1/2) *2*(2**m-1)
        #n3 = 8*Gamma**2 * ( 2**(m+1)*np.pi**3*Lambda_A**3/eps_tot_3  )**(1/2) *2*(2**m-1)

        epsilon_SS = epsilon_S/(r*N) # N comes from Jordan-Wigner, n is the number of U applied

        rost_cost_factor = N*self.tools.c_rotation_synthesis(epsilon_SS)
        
        return rost_cost_factor*r
