import math
import sympy
import numpy as np
from scipy.special import binom

class Taylor_based_methods:

    def __init__(self, tools):

        self.tools = tools

    def configuration_interaction(self, N, eta, alpha, gamma1, K0, K1, K2, epsilon_PEA, epsilon_HS, epsilon_S, epsilon_H, eps_tay, zeta_max_i, phi_max, dphi_max):
        t = 4.7/epsilon_PEA
        x_max = np.log(N * t/ epsilon_HS)
        
        Gamma = binom(eta, 2)*binom(N-eta, 2) + binom(eta,1)*binom(N-eta,1) + 1 # = d
        Zq = eta
        
        '''
        Warning, we have a circular definition here of delta, mu_M_zeta and r.
        In practice we have to find the smallest value of mu_M_zeta compatible with delta:
        mu_M_zeta \\leq f( epsilon_H / 3K*2 Gamma t mu_M_zeta), with f the np.max defining mu_M_zeta below
        Due to this complication we distribute the error uniformly accross all C-U which is not optimal
        '''

        #TODO calculate r and K as satisfaction problem
        r_sym = sympy.Symbol('r_sym')

        def trial_definition(r):
            K = np.log2(r/epsilon_HS)/np.log2(np.log2(r/epsilon_HS))
            delta = epsilon_H/(3*np.log(r)*K) # delta is the error in calculating a single integral. There are 3K log(r) of them in the simulation,
            mu_M_zeta = np.max([ 
            672*np.pi**2/(alpha**3)*phi_max**4*x_max**5*(np.log(K2*phi_max**4*x_max**5/delta))**6,
            256*np.pi**2/(alpha**3)*Zq*phi_max**2*x_max**2*(np.log(K1*Zq*phi_max**2*x_max**2/delta))**3,
            32*gamma1**2**2/(alpha**3)*phi_max**2*x_max*(np.log(K0*phi_max**2*x_max/delta))**3
            ])
            consequent_r = 2*Gamma*t*mu_M_zeta
            return consequent_r

        r = self.tools.bisection(r_sym, trial_definition(r_sym))
        K = np.log2(r/epsilon_HS)/np.log2(np.log2(r/epsilon_HS))

        # end circular definition


        epsilon_SS = epsilon_S / (2*K*2*3*np.log(r))
        Prepare_beta = (20+24*np.log2(1/epsilon_SS))*K

        mu = ( r/epsilon_H *2*(4*dphi_max + phi_max/x_max)*phi_max**3 * x_max**6 )**6
        n = np.log(mu)/3

        x = sympy.Symbol('x')
        K =  np.ceil(np.log2(r/epsilon_HS) / np.log2( np.log2 (r/epsilon_HS)))
        number_sample = 2*K* 3* 2* int(np.ceil(np.log(r)))
        eps_tay_s = eps_tay / number_sample
        order = max(self.tools.order_find(function = math.sqrt(x), x0 = 1, e = eps_tay_s, xeval = x_max),
                    self.tools.order_find(function = math.exp(zeta_max_i*(x)**2), x0 = 0, e = eps_tay_s, xeval = x_max))

        Sample_w = ( 6*35*n**2*(order-1)*4*N + (189+35*(order-1))*n**2 )*K

        Q_val = 2*Sample_w
        Q_col = 6*(32*eta*np.log2(N) + 24*eta**2 + 16*eta*(eta+1)*np.log2(N))
        
        Select_H = Q_val + 2*Q_col
        Select_V = K*Select_H

        return int(np.ceil(np.log(r)))*3*(2*Prepare_beta + Select_V)

    # Taylorization (babbush2016exponential)
    # Let us know calcula the cost of performing Phase Estimation. 
    # 1.  We have already mentioned that in this case, controlling the direction of the time evolution adds negligible cost. We will also take the unitary $U$ in Phase estimation to be $U_r$. The number of segments we will have to Hamiltonian simulate in the phase estimation protocol is $r \\approx \\frac{4.7}{\\epsilon_{\\text{PEA}}}$.
    # 2. Using oblivious amplitude amplification operator $G$ requires to use $\\mathcal{W}$ three times.
    # 3. Each operator $G$ requires to use Prepare$(\\beta)$ twice and Select$(V)$ once.
    # 4. The cost of Select$(V)$ is bounded in $8N\\lceil \\log_2\\Gamma + 1\\rceil\\frac{K(K+1)(2K+1)}{3}+ 16N K(K+1)$.
    # 5. The cost of Prepare$(\\beta)$ is $(20+24\\log\\epsilon^{-1}_{SS})K$ T gates for the preparation of $\\ket{k}$; and $(10+12\\log\\epsilon^{-1}_{SS})2^{\\lceil \\log \\Gamma \\rceil + 1}K$ T gates for the implementation of the $K$ Prepare$(W)$ circuits. Here notice that $2K$ and $2^{\\lceil \\log \\Gamma \\rceil + 1}K$ rotations up to error $\\epsilon_{SS}$ will be implemented.
    # Remember that 
    # $$ K =  O\\left( \\frac{\\log(r/\\epsilon_{HS})}{\\log \\log(r/\\epsilon_{HS})} \\right)$$
    # Notice that the $\\Lambda$ parameters comes in the algorithm only implicitly, since we take the evolution time of a single segment to be $t_1 = \\ln 2/\\Lambda$ such that the first segment in Phase estimation has $r = \\frac{\\Lambda t_1}{\\ln 2} = 1$ as it should be. In general, we will need to implement $r \\approx \\frac{4.7}{\\epsilon_{PEA}}$. However, since $\\epsilon_{PEA}$ makes reference to $H$ and we are instead simulating $H \\ln 2/ \\Lambda$, we will have to calculate the eigenvalue to precision $\\epsilon \\ln 2/ \\Lambda$; so it is equivalently to fixing an initial time $t_1$ and running multiple segments in each of the $U$ operators in Phase Estimation.
    def taylor_naive(self, x, Lambd, Gamma, N):

        # workaround to ensure that the optimizer never check negative values of errors
        # if any error is lower or equal to 0, the cost is penalized to a maximum value
        if not all(x_i > 0 for x_i in x): return self.tools.config_variables['maximum_cost']

        epsilon_PEA = x[0]
        epsilon_HS = x[1]
        epsilon_S = x[2]

        r = 4.7*Lambd / (epsilon_PEA*np.log(2)) # The simulated time
        K_list = []
        
        for m_j in range(0, int(np.ceil(np.log(r)))):
            
            t_j = 2**m_j
            epsilon_HS_mj = epsilon_HS / r * 2**m_j
        
            K = np.ceil(np.log2(t_j/epsilon_HS_mj) / np.log2( np.log2 (t_j/epsilon_HS_mj)))
            K_list.append(K)
            
        result = 0
        epsilon_SS = epsilon_S /(np.sum([3*2*(K*2**(np.ceil(np.log2(Gamma)+1)) + 2*K) for K in K_list]))
            
        for m_j in range(0, int(np.ceil(np.log(r)))):
            
            t_j = 2**m_j
            epsilon_HS_mj = epsilon_HS / r * t_j
        
            K = np.ceil(np.log2(t_j/epsilon_HS_mj) / np.log2( np.log2 (t_j/epsilon_HS_mj)))
            Select_V = 8*N*np.ceil(np.log2(Gamma) +1)*K*(K+1)*(2*K+1)/3 + 16*N*K*(K+1)

            Prepare_beta_1 = (20+24*np.log2(1/epsilon_SS))*K
            Prepare_beta_2 = (10+12*np.log2(1/epsilon_SS))*K*2**(np.ceil(np.log2(Gamma)+1))
            Prepare_beta = Prepare_beta_1 + Prepare_beta_2
            
            result += 3*(2*Prepare_beta + Select_V)

        return result

    def taylor_on_the_fly(self, Gamma, N, phi_max, dphi_max, epsilon_PEA, epsilon_HS, epsilon_S, epsilon_H, zeta_max_i, eps_tay):
        '''
        Error terms 
        eps_PEA: Phase estimation
        eps_HS: the truncation of K
        eps_S: gate synthesis
        eps_H: discretization of integrals
        eps_taylor: truncation of taylor series to order o
        '''

        t = 4.7/epsilon_PEA
        x_max = np.log(N * t/ epsilon_H)
        
        lambd = Gamma*phi_max**4 * x_max**5
        r = lambd* t / np.log(2)
        
        K_list = []
        
        for m_j in range(0, int(np.ceil(np.log(r)))):
            
            t_j = 2**m_j
            epsilon_HS_mj = epsilon_HS / r * 2**m_j
        
            K = np.ceil(np.log2(t_j/epsilon_HS_mj) / np.log2( np.log2 (t_j/epsilon_HS_mj)))
            K_list.append(K)

        epsilon_SS = epsilon_S /np.sum([3*2*(2*K) for K in K_list])
    
        K = np.ceil(np.log2(r/epsilon_HS) / np.log2( np.log2 (r/epsilon_HS)))
        # We distribute the error between all C-U in phase estimation uniformly
        eps_tay_s = eps_tay/((6+2)*K*np.log(r)*3*2)

        x = sympy.Symbol('x')
        order = max(self.tools.order_find(function = math.sqrt(x), x0 = 1, e = eps_tay_s, xeval = x_max),
                    self.tools.order_find(function = math.exp(zeta_max_i*(x)**2), x0 = 0, e = eps_tay_s, xeval = x_max))
        
        result = 0
        
        for m_j in range(0, int(np.ceil(np.log(r)))):
            
            t_j = 2**m_j
            epsilon_HS_mj = epsilon_HS / r * 2**m_j
        
            K = np.ceil(np.log2(t_j/epsilon_HS_mj) / np.log2( np.log2 (t_j/epsilon_HS_mj)))
        
            mu = ( 3*K*2*r/epsilon_H *2*(4*dphi_max + phi_max/x_max)*phi_max**3 * x_max**6 )**6
            n = np.log(mu)/3

            Select_V = 8*N*np.ceil(np.log2(Gamma) +1)*K*(K+1)*(2*K+1)/3 + 16*N*K*(K+1)

            Prepare_beta_1 = (20+24*np.log2(1/epsilon_SS))*K

            Prepare_beta_2 = ( 6*35*n**2*(order-1)*4*N + (252+70*(order-1))*n**2 )*K

            Prepare_beta = Prepare_beta_1 + Prepare_beta_2
            
            result += 3*(2*Prepare_beta + Select_V)
            
        return result