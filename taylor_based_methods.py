import math
import sympy
import numpy as np
from scipy.special import binom
import scipy

class Taylor_based_methods:

    def __init__(self, tools):

        self.tools = tools

    def configuration_interaction(self, N, eta, alpha, gamma1, gamma2, epsilon_PEA, epsilon_HS, epsilon_S, epsilon_H, eps_tay, zeta_max_i, phi_max, dphi_max):
        '''
        gamma1, gamma2, alpha are defined in D9 and D8
        '''
        
        K0 = 26*gamma1/alpha**2 + 8*np.pi*gamma2/alpha**3 + 32*np.sqrt(3)*gamma1*gamma2 # eq D14a
        K1 = 8*np.pi**2/alpha**3*(alpha + 2) + 1121*(8*gamma1 + np.sqrt(2))             # eq D14b
        K2 = 128*np.pi/alpha**6*(alpha + 2) + 2161*np.pi**2*(20*gamma1 + np.sqrt(2))    # eq D14c
        
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

        # min mu_M_zeta such that it is larger or equal to the bound given in eq D13
        def mu_M_zeta_bound_calc(mu_M_zeta):
            r = 2*Gamma*t*mu_M_zeta
            K = np.log2(r/epsilon_HS)/np.log2(np.log2(r/epsilon_HS))
            delta = epsilon_H/(3*r*K)   # delta is the error in calculating a single integral. There are 3K*r of them in the simulation, 
                                        # as r segments are simulated, for a total time of t #todo: this is probably wrong and need 3K r ??

            mu_M_zeta_bound = np.max([ 
                672*np.pi**2/(alpha**3)*phi_max**4*x_max**5*(np.log(K2*phi_max**4*x_max**5/delta))**6,
                256*np.pi**2/(alpha**3)*Zq*phi_max**2*x_max**2*(np.log(K1*Zq*phi_max**2*x_max**2/delta))**3,
                32*gamma1**2**2/(alpha**3)*phi_max**2*x_max*(np.log(K0*phi_max**2*x_max/delta))**3
            ])
            return mu_M_zeta_bound

        # Nonlinear constraint mu_M_zeta >= mu_M_zeta_bound
        nconstraint = scipy.optimize.NonlinearConstraint(fun = lambda mu_M_zeta: mu_M_zeta - mu_M_zeta_bound_calc(mu_M_zeta), lb = 0, ub = +np.inf, keep_feasible = True)

        result = scipy.optimize.minimize(fun = lambda mu_M_zeta: mu_M_zeta, x0 = 1e4, constraints = [nconstraint], tol = 10, options = {'maxiter': 50}, method='COBYLA') # Works with COBYLA, but not with SLSQP (misses the boundaries) or trust-constr (oscillates)

        mu_M_zeta = float(result['x'])
        r = 2*Gamma*t*mu_M_zeta
        K = np.log2(r/epsilon_HS)/np.log2(np.log2(r/epsilon_HS))

        # end circular definition

        epsilon_SS = epsilon_S / (2*K*2*3*r)
        rot_synt = self.tools.rotation_synthesis(epsilon_SS) #todo: create a function that computes(10+12*np.log2(1/epsilon_SS))
        Prepare_beta = 2*rot_synt*K

        mu = ( r/epsilon_H *2*(4*dphi_max + phi_max/x_max)*phi_max**3 * x_max**6 )**6
        n = np.log(mu)/3

        x = sympy.Symbol('x')
        K =  np.ceil(np.log2(r/epsilon_HS) / np.log2( np.log2 (r/epsilon_HS)))
        number_sample = 2*K* 3* 2* r
        eps_tay_s = eps_tay / number_sample
        order = max(self.tools.order_find(function = math.sqrt(x), x0 = 1, e = eps_tay_s, xeval = x_max),
                    self.tools.order_find(function = math.exp(zeta_max_i*(x)**2), x0 = 0, e = eps_tay_s, xeval = x_max))

        Sample_w = ( 6*35*n**2*(order-1)*4*N + (189+35*(order-1))*n**2 )*K

        Q_val = 2*Sample_w
        Q_col = 6*(32*eta*np.log2(N) + 24*eta**2 + 16*eta*(eta+1)*np.log2(N))
        
        Select_H = Q_val + 2*Q_col
        Select_V = K*Select_H

        return r*3*(2*Prepare_beta + Select_V)

    # Taylorization (babbush2016exponential)
    # Let us know calcula the cost of performing Phase Estimation. 
    # 1.  We have already mentioned that in this case, controlling the direction of the time evolution adds negligible cost. We will also take the unitary $U$ in Phase estimation to be $U_r$. The number of segments we will have to Hamiltonian simulate in the phase estimation protocol is $r \\approx \\frac{4.7}{\\epsilon_{\\text{PEA}}}$.
    # 2. Using oblivious amplitude amplification operator $G$ requires to use $\\mathcal{W}$ three times.
    # 3. Each operator $G$ requires to use Prepare$(\\beta)$ twice and Select$(V)$ once.
    # 4. The cost of Select$(V)$ is bounded in $8N\\lceil \\log_2\\Gamma + 1\\rceil\\frac{K(K+1)(2K+1)}{3}+ 16N K(K+1)$.
    # 5. The cost of Prepare$(\\beta)$ is $(20+24\\log\\epsilon^{-1}_{SS})K$ T gates for the preparation of $\\ket{k}$; and $(10+12\\log\\epsilon^{-1}_{SS})2^{\\lceil \\log \\Gamma \\rceil + 1}K$ T gates for the implementation of the $K$ Prepare$(W)$ circuits. Here notice that $2K$ and $2^{\\lceil \\log \\Gamma \\rceil + 1}K$ rotations up to error $\\epsilon_{SS}$ will be implemented.
    # Remember that 
    # $$ K =  O\\left( \\frac{\\log(r/\\epsilon_{HS})}{\\log \\log(r/\\epsilon_{HS})} \\right)$$
    # Notice that the $\\lambda$ parameters comes in the algorithm only implicitly, 
    # since we take the evolution time of a single segment to be $t_1 = \\ln 2/\\lambda$ such that the first segment in Phase estimation has $r = \\frac{\\lambda t_1}{\\ln 2} = 1$ as it should be. 
    # In general, we will need to implement $r \\approx \\frac{4.7}{\\epsilon_{PEA}}$. However, since $\\epsilon_{PEA}$ makes reference to $H$ and we are instead simulating $H \\ln 2/ \\lambda$, 
    # we will have to calculate the eigenvalue to precision $\\epsilon \\ln 2/ \\lambda$; so it is equivalently to fixing an initial time $t_1$ and running multiple segments in each of the $U$ operators in Phase Estimation.
    def taylor_naive(self, epsilons, lambd, Gamma, N):

        epsilon_PEA = epsilons[0]
        epsilon_HS = epsilons[1]
        epsilon_S = epsilons[2]
        
        r = 4.7*lambd / (epsilon_PEA*np.log(2)) # The simulated time

        epsilon_HS_mj = epsilon_HS / r
    
        K = np.ceil(np.log2(1/epsilon_HS_mj) / np.log2( np.log2 (1/epsilon_HS_mj)))
        arb_state_synt = self.tools.arbitrary_state_synthesis(np.ceil(np.log2(Gamma))) #todo: 2**(np.ceil(n+1))
        epsilon_SS = epsilon_S /(3*2*(K*arb_state_synt + 2*K)*r ) # 3 from AA, 2 for for Prepare and Prepare^+, then Prepare_beta_1 and Prepare_beta_2, finally r

        Select_H = 16*(np.ceil(np.log2(Gamma) +1)+3)* 2**4 *N
        Select_V = Select_H * K

        rot_synt = self.tools.rotation_synthesis(epsilon_SS) #todo: create a function that computes(10+12*np.log2(1/epsilon_SS))
        Prepare_beta_1 = 2*rot_synt*K
        Prepare_beta_2 = rot_synt*K*arb_state_synt
        Prepare_beta = Prepare_beta_1 + Prepare_beta_2
        
        return 3*(2*Prepare_beta + Select_V)*r # 3 from AA, 2 Prepare_beta for Prepare and Prepare^+

    def taylor_on_the_fly(self, Gamma, N, phi_max, dphi_max, epsilon_PEA, epsilon_HS, epsilon_S, epsilon_H, eps_tay, zeta_max_i, J):
        '''
        Error terms 
        eps_PEA: Phase estimation
        eps_HS: the truncation of K
        eps_S: gate synthesis
        eps_H: discretization of integrals
        eps_taylor: truncation of taylor series to order o

        zeta_max_i: maximum nuclear charge
        J: number of atoms
        '''
        d = 6 # Number of Gaussians per basis function

        t = 4.7/epsilon_PEA
        x_max = np.log(N * t/ epsilon_H)
        
        lambd = Gamma*phi_max**4 * x_max**5
        r = lambd* t / np.log(2)

        K = np.ceil(np.log2(r/epsilon_HS) / np.log2( np.log2 (r/epsilon_HS)))

        epsilon_SS = epsilon_S /(r*3*2*(2*K)) # 3 from AA, 2 Prepare_beta for Prepare and Prepare^+, 2K T gates in the initial theta rotations
        # We distribute the error between all C-U in phase estimation uniformly
        number_of_taylor_expansions = (((4+2+2)*d*N + (J+1))*K*2*3*r) #4+2+2 = two_body + kinetic + external_potential
        eps_tay_s = eps_tay/number_of_taylor_expansions
        x = sympy.Symbol('x')
        order = max(self.tools.order_find(function = math.sqrt(x), x0 = 1, e = eps_tay_s, xeval = x_max),
                    self.tools.order_find(function = math.exp(zeta_max_i*(x)**2), x0 = 0, e = eps_tay_s, xeval = x_max))
        
        mu = ( r*3*2*K/epsilon_H *2*(4*dphi_max + phi_max/x_max)*phi_max**3 * x_max**6 )**6
        n = np.ceil(np.ceil(np.log2(mu))/3) #each coordinate is a third
        M = lambd*r*3*2*K/epsilon_H

        sum = self.tools.sum_cost(n) #todo: 4*n
        mult = self.tools.multiplication_cost(n) #todo: 21*n**2
        div = self.tools.divide_cost(n) #todo: 14n**2+7*n

        tay = order*sum + (order-1)*(mult + div)

        Qphi = N*d((3*sum) + (3*mult +2*sum) + (mult) + tay + (3*mult)) #In parenthesis each step in the list
        Qnabla = Qphi + N*d*(4*sum+mult+div)
        R = 2*mult + sum + tay
        xi = 3*sum
        #Qphi = N*d*(35*n**2*order + 91*n**2+13*n*order + 13*n)
        #Qnabla = N*d*(35*n**2*order + 126*n**2+13*n*order + 32*n)
        #R = 35*n**2*order + 6*n**2 + 13*n*order - 3*n
        #xi = 12*n 
    
        two_body = xi + 4*Qphi + R + 4*mult
        kinetic = Qphi + Qnabla + mult
        external_potential = 2*Qphi + J*R + J*mult + (J-1)*sum + xi*J
        sample = two_body + kinetic + external_potential

        # Notice the change of n here: it is the size of register |m>
        n = np.ceil(np.log2(M))
        sum = self.tools.sum_cost(n) #todo: 4*n
        mult = self.tools.multiplication_cost(n) #todo: 21*n**2
        div = self.tools.divide_cost(n) #todo: 14n**2+7*n
        comp = self.tools.compare_cost(max(np.ceil(np.log2(M)),np.ceil(np.log2(mu)))) #todo: 8*n

        Ri = 2*(mult + 3*sum + comp) #For the comparison operation. The rotation itself is Clifford, as it is a C-R(pi/2)

        rot_synt = self.tools.rotation_synthesis(epsilon_SS) #todo: create a function that computes(10+12*np.log2(1/epsilon_SS))

        Prepare_beta_1 = 2*rot_synt*K
        Prepare_beta_2 = ( 2*sample + Ri )*K
        Prepare_beta = Prepare_beta_1 + Prepare_beta_2

        Select_H = 16*(np.ceil(np.log2(Gamma) +1)+3)* 2**4 *N
        Select_V = Select_H * K

        result = 3*(2*Prepare_beta + Select_V)*r

        return result