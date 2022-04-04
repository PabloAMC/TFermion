import math
import sympy
import numpy as np
from scipy.special import binom
import scipy

class Taylor_based_methods:

    def __init__(self, tools):

        self.tools = tools

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
    def taylor_naive(self, epsilons, lambda_value, Gamma, N):

        epsilon_PEA = epsilons[0]
        epsilon_HS = epsilons[1]
        epsilon_S = epsilons[2]
        
        t = np.pi/epsilon_PEA
        r = np.ceil(t*lambda_value / np.log(2)) # Number of time segments
    
        K = np.ceil( -1  + 2* np.log(2*r/epsilon_HS)/np.log(np.log(2*r/epsilon_HS)+1)) 
        arb_state_synt = self.tools.arbitrary_state_synthesis(4*np.ceil(np.log2(N)))
        epsilon_SS = epsilon_S /(r*3*2*(K*arb_state_synt + 2*K) ) # 3 from AA, 2 for for Prepare and Prepare^+, then Prepare_beta_1 and Prepare_beta_2, finally r

        Select_j = 4*N*self.tools.multi_controlled_not(np.ceil(np.log2(N))+2) + 4*N + N*self.tools.multi_controlled_not(np.ceil(np.log2(N)))
        # We use an accumulator that applies C-Z and upon stop applies the X or Y with phase: The 4 comes from values of q, the N from values of j;
        # the first term applies the X or Y (and phase); the 4N comes from the Toffolis in the C-Z; the third term deactivates the accumulator
        Select_H = 4*Select_j # 4 creation/annihilation operators per H_\gamma
        QPE_adaptation = self.tools.multi_controlled_not(np.ceil(K/2) + 1) 
        Select_V = Select_H * K + QPE_adaptation

        crot_synt = self.tools.c_pauli_rotation_synthesis(epsilon_SS)
        rot_synt = self.tools.pauli_rotation_synthesis(epsilon_SS)
        Prepare_beta_1 = crot_synt*K
        Prepare_beta_2 = rot_synt*K*arb_state_synt
        Prepare_beta = Prepare_beta_1 + Prepare_beta_2

        R = self.tools.multi_controlled_not((K+1)*np.ceil(np.log2(Gamma)) + N) # The prepare qubits and the select qubits (in Jordan-Wigner there are N)
        result = r*(3*(2*Prepare_beta + Select_V) + 2*R)  # 3 from AA, 2 Prepare_beta for Prepare and Prepare^+
        
        return result

    def taylor_on_the_fly(self, epsilons, N, Gamma, phi_max, dphi_max, zeta_max_i, J):
        
        epsilon_PEA = epsilons[0]
        epsilon_HS = epsilons[1]
        epsilon_S = epsilons[2]
        epsilon_H = epsilons[3]
        eps_tay = epsilons[4] 

        '''
        Error terms 
        eps_PEA: Phase estimation
        eps_HS: the truncation of K
        eps_S: gate synthesis
        eps_H: discretization of integrals
        eps_taylor: Used for arithmetic operations such as taylor series, babylon algorithm for the sqrt and CORDIC algorithm for cos


        zeta_max_i: maximum nuclear charge
        J: number of atoms
        '''
        d = 6 # Number of Gaussians per basis function

        t = np.pi/epsilon_PEA
        x_max = np.log(N * t/ epsilon_H)* self.tools.config_variables['xmax_mult_factor_taylor'] # eq 68 in the original paper
        
        Vol_max_w_gamma = (2**6*phi_max**4 * x_max**5) # eq 66 in the original article
        lambda_value = Gamma*Vol_max_w_gamma # eq 60 in the original article
        r = np.ceil(lambda_value* t / np.log(2)) 
        K = np.ceil( -1  + 2* np.log(2*r/epsilon_HS)/np.log(np.log(2*r/epsilon_HS)+1))

        # zeta = epsilon_HS /(2*3*K*r*Gamma*Vol); eq 55 in the original article
        M = lambda_value* 2*3*K*r/epsilon_H # = 6*K*r*Gamma*Vol_max_w_gamma/epsilon_H; eq 55 in the original article

        epsilon_SS = epsilon_S /(r*3*2*(2*K)) # 3 from AA, 2 Prepare_beta for Prepare and Prepare^+, 2K T gates in the initial theta rotations

        number_of_taylor_expansions = (((4+2+2)*d*N + (J+1))*K*2*3*r) #4+2+2 = two_body + kinetic + external_potential
        eps_tay_s = eps_tay/number_of_taylor_expansions
        x = sympy.Symbol('x')

        exp_order = self.tools.order_find(lambda x:math.exp(zeta_max_i*(x)**2), e = eps_tay_s, xeval = x_max, function_name = 'exp')
        sqrt_order = self.tools.order_find(lambda x:math.sqrt(x), e = eps_tay_s, xeval = x_max, function_name = 'sqrt')
        
        mu = ( r*3*2*K/epsilon_H *2*(4*dphi_max + phi_max/x_max)*phi_max**3 * x_max**6 )**6
        n = np.ceil(np.ceil(np.log2(mu))/3) #each coordinate is a third

        sum = self.tools.sum_cost(n)
        mult = self.tools.multiplication_cost(n)
        div = self.tools.divide_cost(n)

        tay = exp_order*sum + (exp_order-1)*(mult + div) # For the exp
        babylon = sqrt_order*(div +  sum) # For the sqrt

        Q = N*d*((3*sum) + (3*mult +2*sum) + (mult) + tay + (3*mult)) #In parenthesis each step in the list
        Qnabla = Q + N*d*(4*sum+mult+div)
        R = 2*mult + sum + babylon
        xi = 3*sum

        two_body = xi + 4*Q + R + 4*mult
        kinetic = Q + Qnabla + mult
        external_potential = 2*Q + J*R + J*mult + (J-1)*sum + xi*J
        sample = two_body + (kinetic + external_potential + sum)

        # Notice the change of n here: it is the size of register |m>
        n = np.ceil(np.log2(M))
        sum = self.tools.sum_cost(n)
        mult = self.tools.multiplication_cost(n) 
        div = self.tools.divide_cost(n)
        comp = self.tools.compare_cost(max(np.ceil(np.log2(M)),np.ceil(np.log2(mu))))

        kickback = 2*(mult + 3*sum + comp) #For the comparison operation. The rotation itself is Clifford, as it is a C-R(pi/2)

        crot_synt = self.tools.c_pauli_rotation_synthesis(epsilon_SS)
        Prepare_beta_1 = crot_synt*K
        Prepare_beta_2 = ( 2*sample + kickback )*K
        Prepare_beta = Prepare_beta_1 + Prepare_beta_2

        Select_j = 4*N*self.tools.multi_controlled_not(np.ceil(np.log2(N))+2) + 4*N + N*self.tools.multi_controlled_not(np.ceil(np.log2(N)))
        # The 4 comes from values of q, the N from values of j; the 4N comes from the Toffolis in the C-Z; the third term deactivates the accumulator
        Select_H = 4*Select_j
        QPE_adaptation = self.tools.multi_controlled_not(np.ceil(K/2) + 1) 
        Select_V = Select_H * K + QPE_adaptation

        R = self.tools.multi_controlled_not((K+1)*np.log2(Gamma) + N) # The prepare qubits and the select qubits (in Jordan-Wigner there are N)
        result = r*(3*(2*Prepare_beta + Select_V) + 2*R)

        return result

    def configuration_interaction(self, epsilons, N, eta, alpha, gamma1, gamma2, zeta_max_i, phi_max, J):

        epsilon_PEA = epsilons[0]
        epsilon_HS = epsilons[1]
        epsilon_S = epsilons[2]
        epsilon_H = epsilons[3]
        eps_tay = epsilons[4]

        '''
        gamma1, gamma2, alpha are defined in 28, 29 and 30 of the original paper https://iopscience.iop.org/article/10.1088/2058-9565/aa9463/meta
        '''
        d = 6 ## THIS IS SORT OF AN HYPERPARAMETER: THE NUMBER OF GAUSSIANS PER BASIS FUNCTION
        
        K0 = 26*gamma1/alpha**2 + 8*np.pi*gamma2/alpha**3 + 32*np.sqrt(3)*gamma1*gamma2 # eq 37 in original article
        K1 = 8*np.pi**2/alpha**3*(alpha + 2) + 1121*(8*gamma1 + np.sqrt(2))             # eq 41 in original article
        K2 = 128*np.pi/alpha**6*(alpha + 2) + 2161*np.pi**2*(20*gamma1 + np.sqrt(2))    # eq 45 in original article
        
        t = np.pi/epsilon_PEA
        x_max = 1 # Default units are Angstroms. See https://en.wikipedia.org/wiki/Atomic_radius and https://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page)
        
        Gamma = binom(eta, 2)*binom(N-eta, 2) + binom(eta,1)*binom(N-eta,1) + 1 # = d
        Zq = eta
        
        '''
        Warning, we have a circular definition here of delta, mu_M_zeta and r.
        In practice we compute the equality value r given by the lemmas in the paper:
        r ~= r_bound_calc(r)
        '''

        def r_bound_calc(r):
            K = np.ceil( -1  + 2* np.log(2*r/epsilon_HS)/np.log(np.log(2*r/epsilon_HS)+1)) 
            delta = epsilon_H/(2*3*r*K)   # delta is the error in calculating a single integral. There are 2*3K*r of them in the simulation, 
                                        # as r segments are simulated, for a total time of t

            mu_M_zeta_bound = np.max([ 
                672*np.pi**2/(alpha**3)*phi_max**4*x_max**5*(np.log(K2*phi_max**4*x_max**5/delta))**6,
                256*np.pi**2/(alpha**3)*Zq*phi_max**2*x_max**2*(np.log(K1*Zq*phi_max**2*x_max**2/delta))**3,
                32*gamma1**2/(alpha**3)*phi_max**2*x_max*(np.log(K0*phi_max**2*x_max/delta))**3
            ])   #This bound is so because Lemmas 1-3 are bounding aleph_{\gamma,\rho}. Taking the definition of M, it is clear.

            r_bound = 2*Gamma*t*mu_M_zeta_bound/np.log(2)
            return r_bound

        result = scipy.optimize.minimize(fun = lambda logr: (logr - np.log(r_bound_calc(np.exp(logr))))**2, x0 = 25, tol = .05, options = {'maxiter': 5000}, method='COBYLA') # Works with COBYLA, but not with SLSQP (misses the boundaries) or trust-constr (oscillates)
        logr = np.ceil(result['x'])
        r = np.exp(logr)

        #bound = r_bound_calc(r) #This should be close to each r, relatively speaking
        #r_alt = r_bound_calc(Gamma*t) #Alternative and less accurate way of computing the result

        K = np.ceil( -1  + 2* np.log(2*r/epsilon_HS)/np.log(np.log(2*r/epsilon_HS)+1)) 

        delta = epsilon_H/(2*3*r*K)
        
        mu_M_zeta = np.max([ 
            672*np.pi**2/(alpha**3)*phi_max**4*x_max**5*(np.log(K2*phi_max**4*x_max**5/delta))**6,
            256*np.pi**2/(alpha**3)*Zq*phi_max**2*x_max**2*(np.log(K1*Zq*phi_max**2*x_max**2/delta))**3,
            32*gamma1**2/(alpha**3)*phi_max**2*x_max*(np.log(K0*phi_max**2*x_max/delta))**3
        ])

        log2mu = np.max([
            6*(np.log2(K2*phi_max**4*x_max**5) + np.log2(1/delta) + 7*np.log2(1/alpha*(np.log(K2*phi_max**4*x_max**5)+np.log(1/delta)))),
            3*(np.log2(K1*Zq*phi_max**2*x_max**2)+np.log2(1/delta) + 4*np.log2(2/alpha*(np.log(K1*Zq*phi_max**2*x_max**2)+np.log(1/delta)))),
            3*(np.log2(K0*phi_max**2*x_max)+np.log2(1/delta) +4*np.log2 (2/alpha*(np.log(K0*phi_max**2*x_max)+np.log(1/delta))))
        ])
        #zeta = epsilon_H/(r*Gamma*mu*3*2*K)
        log2M = np.ceil(np.log2(mu_M_zeta)+ np.log2(3*2*K*r*Gamma)+ np.log2(1/epsilon_H)) #M = mu_M_zeta*/(mu*zeta)

        epsilon_SS = epsilon_S / (r*3*2*(2*K)) # 3 from AA, 2 Prepare_beta for Prepare and Prepare^+, 2K T gates in the initial theta rotations
        crot_synt = self.tools.c_pauli_rotation_synthesis(epsilon_SS)
        Prepare_beta = crot_synt*K

        #### Qval cost computation
        n = np.ceil(log2mu/3) #each coordinate is a third
        x = sympy.Symbol('x')

        number_of_taylor_expansions = (((2*4+2+2)*d*N + (J+1))*K*2*3*r) #2*4+2+2 = 2*two_body + kinetic + external_potential
        eps_tay_s = eps_tay/number_of_taylor_expansions

        exp_order = self.tools.order_find(lambda x:math.exp(zeta_max_i*(x)**2), function_name = 'exp', e = eps_tay_s, xeval = x_max)
        sqrt_order = self.tools.order_find(lambda x:math.sqrt(x), function_name = 'sqrt', e = eps_tay_s, xeval = x_max)

        sum = self.tools.sum_cost(n)
        mult = self.tools.multiplication_cost(n)
        div = self.tools.divide_cost(n)

        tay = exp_order*sum + (exp_order-1)*(mult + div) # For the exp
        babylon = sqrt_order*(div +  sum) # For the sqrt

        Q = N*d*((3*sum) + (3*mult +2*sum) + (mult) + tay + (3*mult)) #In parenthesis each step in the list
        Qnabla = Q + N*d*(4*sum+mult+div)
        R = 2*mult + sum + babylon
        xi = 3*sum
    
        two_body = xi + 4*Q + R + 4*mult
        kinetic = Q + Qnabla + mult
        external_potential = 2*Q + J*R + J*mult + (J-1)*sum + xi*J
        sample_2body = 2*two_body + sum
        sample_1body =  kinetic + external_potential + sum

        comp = self.tools.compare_cost(max(np.ceil(log2M),np.ceil(log2mu)))
        kickback = 2*comp

        Q_val = 2*(sample_2body + sample_1body) + kickback

        ### Qcol cost computation

        # There will be eta registers with log2(N) qubits each
        compare = self.tools.compare_cost(np.ceil(np.log2(N)))
        sort = eta*(4 + compare) # 4 for the c-swap and one comparison
        check = self.tools.multi_controlled_not(eta*np.ceil(np.log2(N)))
        sum = self.tools.sum_cost(np.ceil(np.log2(N)))

        find_alphas = 2* eta*(4*sum + check + sort + compare) #The 2 is because if it fails we have to reverse the computation
        find_gammas_2y4 = 2*(3*sum + check+ sort+ compare +3*4) + find_alphas  # The 3*4 is the final 3 Toffolis; the 2 is is because if it fails we have to reverse the computation 
        
        Q_col = 2*find_alphas + 2*find_gammas_2y4
        
        Select_H = Q_val + 2*Q_col # +swaps, but they are Clifford
        QPE_adaptation = self.tools.multi_controlled_not(np.ceil(K/2) + 1) 
        Select_V = K*Select_H + QPE_adaptation

        R = self.tools.multi_controlled_not((K+1)*np.ceil(np.log2(Gamma)) + N) # The prepare qubits and the select qubits (in Jordan-Wigner there are N)
        result = r*(3*(2*Prepare_beta + Select_V) + 2*R)

        return result