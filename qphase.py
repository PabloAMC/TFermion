import numpy as np
from itertools import combinations
import scipy
from scipy.optimize import minimize
from scipy.special import binom, gamma
from scipy.integrate import quad, dblquad
from scipy import integrate
import sympy
import math

## The aim of this notebook is to calculate a cost estimation of different methods to calculate the energy of a system with different Phase Estimation protocols
## IMPORTANT: to these cost, we have to add the QFT cost, which is minor, and has an associated error.

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

# We want to minimize the total cost,
# $$3n(10+12\\log \\epsilon^{-1})\\log N  = 3\\frac{27\\pi^2}{2}\\frac{\\lambda^2}{\\delta_E^2 P_f}(10+12\\log \\epsilon^{-1})\\log N$$,
# where $\\epsilon$ is the error of individual rotations $$\\epsilon = \\frac{\\epsilon(j)}{n(j)} = \\frac{\\epsilon_{tot}^2}{4\\pi^2(2^m-1)^2}$$

def calc_qdrift_resources(lambd, N, deltaE = 1e-4, P_failure = .1):
    n = ((27*np.pi**2/2)*(lambd/deltaE)**2) / P_failure**3

    delta = deltaE/(2*lambd)
    q = np.log2(1/delta)-1

    pf = 2/3*P_failure
    eps_tot = P_failure/6
    #sanity check
    assert (pf +2*eps_tot)/P_failure == 1
    
    m = q + np.log2(1/(2*pf)+1/2)
    
    # Another sanity check. This should coincide
    eps_tot_2 = 4*(np.pi*2**m-1)**2/n
    print(eps_tot,eps_tot_2)
    
    # error in individual rotations
    eps = (eps_tot/(2*np.pi*(2**m-1)))**2

    rost_cost_factor = 3*(10+12*np.log(1/eps))*np.log(N)
    print('eps',eps)
    
    return rost_cost_factor*n

# For the randomised Hamiltonian approach, the equations are similar. However, now $p_f = 3/4P_f$ and ,
# $$n = 8\\Gamma^2\\left(\\frac{ \\pi^3 \\Lambda^3}{8\\delta_E^3}\\right)^{1/2}\\left(\\frac{1+p_f}{p_f}\\right)^{3/2}\\frac{1}{\\epsilon_{tot}^{1/2}} = 4.35\\sqrt{8}\\pi^{3/2}\\Gamma^2 \\frac{\\Lambda^{3/2}}{\\delta_E^{3/2}P_f^2}$$ 
def calc_rand_ham_resources(Lambd, lambd, Gamma, N, deltaE = 1e-4, P_failure = .1):
    n = 4.35*np.sqrt(8)*(np.pi*Lambd/deltaE)**(3/2) *(Gamma/ P_failure)**2
    print('n',n)

    # error in individual rotations
    Lambda_A = Lambd/(2*lambd)
    delta = deltaE/(2*lambd)
    q = np.log2(1/delta)-1
    
    pf = 3/4*P_failure
    eps_tot = P_failure/8
    #sanity check
    assert (pf +2*eps_tot)/P_failure == 1
    
    m = q + np.log2(1/(2*pf)+1/2)
    
    n1 = 8*Gamma**2 * ( 2**(m+1)*np.pi**3*Lambda_A**3/eps_tot  )**(1/2) *2*(2**m-1)
    print('n1',n1)
    
    # Another sanity check. This should coincide
    eps_tot_2 = ( 8*Gamma**2* (np.pi*Lambd/(2*deltaE))**(3/2)* ((1+pf)/pf)**(3/2) /n1)**2
    eps_tot_3 = 1/ ( 4.35* (1/P_failure)**2  *  (pf/(1+pf))**(3/2)  )**2
    print(eps_tot,eps_tot_2,eps_tot_3)
    
    n2 = 8*Gamma**2 * ( 2**(m+1)*np.pi**3*Lambda_A**3/eps_tot_2  )**(1/2) *2*(2**m-1)
    print('n2',n2)
    
    n3 = 8*Gamma**2 * ( 2**(m+1)*np.pi**3*Lambda_A**3/eps_tot_3  )**(1/2) *2*(2**m-1)
    print('n3',n3)

    # Esto probablemente esté mal:
    eps = 1/4*(eps_tot/(np.pi*2**m*Lambda_A))**(3/2)

    rost_cost_factor = 3*(10+12*np.log(1/eps))*np.log(N)
    print('eps',eps)
    
    return rost_cost_factor*n

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
def Taylor_naive(Lambd, Gamma, N, epsilon_PEA = .4*eps_tot, epsilon_HS = .2*eps_tot, epsilon_S = .4*eps_tot):
    
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

def Taylor_on_the_fly(Gamma, N, phi_max, dphi_max, zeta_max_i, epsilon_PEA = .4*eps_tot, epsilon_HS = .1*eps_tot, epsilon_S = .4*eps_tot, epsilon_H = .1*eps_tot, order = 10):
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
    
    # We distribute the error between all C-U in phase estimation uniformly
    eps_tay_m_j = eps_tay/((6+2)*np.max(K_list)*np.log(r)*3*2)

    x = sympy.Symbol('x')
    order = max(order_find(function = math.sqrt(x), x0 = 1, e = eps_tay_m_j, xeval = x_max),
                order_find(function = math.exp(max_zeta_i*(x)**2), x0 = 0, e = eps_tay_m_j, xeval = x_max))
    
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

# Taylor approximation at x0 of the function 'function'
def taylor(function,x0,n):
    i = 0
    p = 0
    while i <= n:
        p = p + (function.diff(x,i).subs(x,x0))/(factorial(i))*(x-x0)**i
        i += 1
    return p

#print(taylor(sympy.sqrt(x), 1, 5))#.subs(x,1).evalf())

def order_find(function, x0, e, xeval):
    
    x = sympy.Symbol('x')
    
    order = 0
    te = 1
    zeta = np.linspace(x0,xeval,20)

    while te > e:# or order < 10:
        order += 1
        #for z in zeta:
            #print(taylor_err(f, x0, order, z).subs(x,xeval).evalf())
        te = np.max([np.abs(taylor_err(function, x0, order, z).subs(x,xeval).evalf()) for z in zeta])
        #print('order',order, te,'\')
        
    return order

def factorial(n):
    if n <= 0:
        return 1
    else:
        return n*factorial(n-1)
    
def taylor_err(function,x0,n, z = None):
    if z == None:
        z = x0
    #print('coefficient order',n, function.diff(x,n)/(factorial(n)))#.subs(x,z))
    a = (function.diff(x,n).subs(x,z))/(factorial(n))*(x-x0)**n
    #print('coefficient order',n, (function.diff(x,n).subs(x,z)/(factorial(n))*(x-x0)**n))
    #print('a',a)
    return a


eps_tot = .0125
def error_optimizer(eps_array):
    epsilon_PEA = eps_array[0]
    
    epsilon_S = eps_array[1]
    
    epsilon_HS = eps_tot - eps_array[0] - eps_array[1]
    
    return Taylor_naive(Lambd = 4.07, Gamma =467403, N = (467403)**(1/4),
                 epsilon_PEA = epsilon_PEA, epsilon_HS= epsilon_HS, epsilon_S = epsilon_S)

eps_tot = 0.125
def configuration_interaction(N, eta, alpha, gamma1, K0, K1, K2, epsilon_PEA = .4*eps_tot, epsilon_HS = .1*eps_tot, epsilon_S = .4*eps_tot, epsilon_H = .1*eps_tot):
    t = 4.7/epsilon_PEA
    x_max = np.log(N * t/ epsilon_HS)
    
    Gamma = binom(eta, 2)*binom(N-eta, 2) + binom(eta,1)*binom(N-eta,1) + 1 # = d
    Zq = eta
    
    '''
    Warning, we have a circular definition here of delta, mu_M_zeta and r.,
    In practice we have to find the smallest value of mu_M_zeta compatible with delta:,
    mu_M_zeta \\leq f( epsilon_H / 3K*2 Gamma t mu_M_zeta), with f the np.max defining mu_M_zeta below,
    Due to this complication we distribute the error uniformly accross all C-U which is not optimal,
    '''

    delta = epsilon_H/(3*np.log(r)*K) # delta is the error in calculating a single integral. There are 3K log(r) of them in the simulation,

    # This is an upper bound, not an equality!!!
    mu_M_zeta = np.max([ 
        672*np.pi**2/(alpha**3)*phi_max**4*x_max**5*(np.log(K2*phi_max**4*x_max**5/delta))**6,
        256*np.pi**2/(alpha**3)*Zq*phi_max**2*x_max**2*(np.log(K1*Zq*phi_max**2*x_max**2/delta))**3,
        32*gamma1**2**2/(alpha**3)*phi_max**2*x_max*(np.log(K0*phi_max**2*x_max/delta))**3
    ])
    
    r = 2*Gamma*t*mu_M_zeta
    K = np.log2(r/epsilon_HS)/np.log2(np.log2(r/epsilon_HS))
    epsilon_SS = epsilon_S / (2*K*2*3*np.log(r))
    Prepare_beta = (20+24*np.log2(1/epsilon_SS))*K
        
    mu = ( r/epsilon_H *2*(4*dphi_max + phi_max/x_max)*phi_max**3 * x_max**6 )**6
    n = np.log(mu)/3
    Sample_w = ( 6*35*n**2*(order-1)*4*N + (189+35*(order-1))*n**2 )*K

    Q_val = 2*Sample_w
    Q_col = 6*(32*eta*np.log2(N) + 24*eta**2 + 16*eta*(eta+1)*np.log2(N))
    
    Select_H = Q_val + 2*Q_col
    Select_V = K*Select_H
    
    return np.log(r)*3*(2*Prepare_beta + Select_V)

# Low depth quantum simulation of materials (babbush2018low) Trotter
def low_depth_trotter(N, Omega, eps_PEA, eps_HS, eps_S):
    def f(x, y):
        return 1/(x**2 + y**2)
    def I(N0):
        return integrate.nquad(f, [[1, N0],[1, N0]])[0]
    
    t = 4.7/eps_PEA
    sum_1_nu = 4*np.pi(np.sqrt(3)*N**(1/3)/2 - 1) + 3 - 3/N**(1/3) + 3*I(N**(1/3))
    max_V = eta**2/(2*np.pi*Omega**(1/3))*sum_1_nu
    max_U = eta**2/(np.pi*Omega**(1/3))*sum_1_nu
    nu_max = 3*(N**(1/3))**2
    max_T = 2*np.pi**2*eta/(Omega**(2/3))* nu_max
    
    r = np.sqrt(2*t**3/eps_HS *(max_T**2*(max_U + max_V) + max_T*(max_U + max_V)**2))

    eps_SS = eps_S/(2*N +N*(N-1) + N*np.log(N/2) + 8*N)
    
    exp_UV_cost = (4*N**2 + 4*N)*np.log(1/eps_SS)
    FFFT_cost = (2 + 4*np.log(1/eps_SS))*n*np.log(N) + 4*N*np.log(1/eps_SS)
    exp_T_cost = 32*N*np.log(1/eps_SS)
    
    return r*(exp_UV_cost + FFFT_cost + exp_T_cost)

# Low depth quantum simulation of materials (babbush2018low) Taylor
def low_depth_taylor(N, lambd, Lambd, eps_PEA, eps_HS, eps_S, Ham_norm):
    '''To be used in plane wave basis'''
    t = 4.7/eps_PEA
    r = t*Lambda/np.log(2)
    
    K_list = []
    
    for m_j in range(0, int(np.ceil(np.log(r)))):
        
        t_j = 2**m_j
        epsilon_HS_mj = epsilon_HS / r * 2**m_j
    
        K = np.ceil(np.log2(t_j/epsilon_HS_mj) / np.log2( np.log2 (t_j/epsilon_HS_mj)))
        K_list.append(K)
        
    epsilon_SS = epsilon_S /np.sum([3*2*(2*K) for K in K_list]) # The extra two is because Uniform requires 2 Rz gates
    
    mu = np.ceil(np.log(2*np.sqrt(2)*Lambdd/eps_PEA) + np.log(1 + eps_PEA/(8*lambd)) + np.log(1 - (Ham_norm/lambd)**2))
    
    result = 0
    
    for m_j in range(0, int(np.ceil(np.log(r)))):
        
        t_j = 2**m_j
        epsilon_HS_mj = epsilon_HS / r * 2**m_j
    
        K = np.ceil(np.log2(t_j/epsilon_HS_mj) / np.log2( np.log2 (t_j/epsilon_HS_mj)))
    
        prepare_beta = K*(6*N+40*np.log(N)+16*np.log(1/epsilon_SS) + 10*mu)
        select_V = K*(12*N+8*np.log(N))
        
        result += 3*(2*prepare_beta + select_V)
    
    return result

# Low depth quantum simulation of materials (babbush2018low) On-the fly
def low_depth_taylor(N, lambd, Omega, eps_PEA, eps_HS, eps_S, Ham_norm, J):
    '''To be used in plane wave basis
    J: Number of atoms
    '''
    Lambd = (2*eta+1)*N**3 / (2*Omega**(1/3)*np.pi)
    t = 4.7/eps_PEA
    r = t*Lambd/np.log(2)
    
    mu = np.ceil(np.log(2*np.sqrt(2)*Lambdd/eps_PEA) + np.log(1 + eps_PEA/(8*lambd)) + np.log(1 - (Ham_norm/lambd)**2))
    #K_list = []
    
    #epsilon_SS = epsilon_S /np.sum([3*2*(2*K) for K in K_list])
        
    x = sympy.Symbol('x')
    order = order_find(function = cos(x), x0 = 1, e = e, xeval = x_max)
    sample_w = 70*np.log(N)**2 + 29* np.log(N) + (21+14)*order/2*np.log(N)**2 + 2*order*np.log(N) + J*(35*order/2 + 63 + 2*order/np.log(N))*np.log(N)**2
    kickback = 32*np.log(mu)

    result = 0
    
    for m_j in range(0, int(np.ceil(np.log(r)))):
        
        t_j = 2**m_j
        epsilon_HS_mj = epsilon_HS / r * 2**m_j
    
        #K_list.append(K)
    
        K = np.ceil(np.log2(t_j/epsilon_HS_mj) / np.log2( np.log2 (t_j/epsilon_HS_mj)))
    
        prepare_W = 2*sample_w + kickback
        prepare_beta = K*prepare_W
        select_H = (12*N + 8*np.log(N))
        select_V = K*select_H
        
        result += 3*(2*prepare_beta + select_V)
    
    return result

## Linear T complexity (babbush2018encoding)
def linear_T(N, lambd, eps_PEA, eps_SS):
    '''To be used in plane wave basis'''
    t = 4.7/eps_PEA
    r = lambd*t
    
    mu = np.ceil(np.log(2*np.sqrt(2)*lambd/eps_PEA) + np.log(1 + eps_PEA/(8*lambd)) + np.log(1 - (Ham_norm/lambd)**2))
    
    eps_SS = eps_S / (r*2*P)
    
    S = 12*N+8*np.log(N)
    P = 6*N + 40*np.log(N)+ 24*np.log(1/eps_SS) + 10*mu
    
    return r*(2*P + S)

## Sparsity and low rank factorization (berry2019qubitization)
def sparsity_low_rank(N, lambd, eps_PEA, eps_SS, L):
    t = 4.7/eps_PEA
    r = lambd*t
    
    mu = np.ceil(np.log(2*np.sqrt(2)*lambd/eps_PEA) + np.log(1 + eps_PEA/(8*lambd)) + np.log(1 - (Ham_norm/lambd)**2))
    d = L(N**2/8 + N/4)
    M = np.log(N**2) + mu
    
    def closest_power(x):
        possible_results = np.floor(np.log2(x)), np.ceil(np.log2(x))
        return min(possible_results, key= lambda z: abs(x-2**z))
    
    kc = 2**closest_power(np.sqrt(d/M))
    ku = 2**closest_power(np.sqrt(d))
    
    QROAM = 4*(np.ceil(d/kc)+4*M*(kc-1)+2*np.ceil(d/ku) + 4*k_u)
    
    Select = (4*N + 4*np.log(N))*4 # The *4 because Toffoli -> T-gates
    
    # 7 times per prepare, we have to use Uniform
    eps_SS = eps_S/ (7*2*r)
    Uniform = 8*np.log(L) + 56*np.log(1/eps_SS) + 52*np.log(N/2) ### Warning, this is in T gates already!!!!
    
    Other_subprepare = mu + np.log(L) + 6*np.log(N/2)
    
    continuous_register = 2*(np.log(N/2))**2 + 3*np.log(N/2)
    
    Prepare = 4*(QROAM + Other_subprepare + continuous_register) + Uniform # The 4 is Toffoli -> T-gates
    
    return r*(2*Prepare + Select)

def interaction_picture(N, Gamma, lambd_T, lambd_U_V, eps_S, eps_HS, eps_PEA):
    '''
    The number of rotations is very large here:
    Each of the r segments can be simulated as e^{-i(U+V)t} T(e^{-i \\int H_I (s) ds})
    - The Time Ordered Dyson series segment is represented by TDS
    - TDS is made of oblivious Amplitude Amplification of TDS_beta: 2x Ref  + 3x TDS_beta
        < TDS_beta is made of COEF DYS_K COEF'
        < DYS_K is made of
            · 4K U operators
            · K Compare and K Swap
            · (3K + 1) ADD operators
            · K HAM-T operators, made of
                > x2 e^{-i(U+V)t}
                > x2 FFFT
                > x2 Prepare
                > Select
    Also, the e^{-i(U+V)t} is
        > x2 FFFT
        > N log 1/eps_SS Phase operators
        > N Multiplications
    '''

    t = 4.7/eps_PEA
    r = lambd_T*t # lambd_T is necessary to take tau = 1
    
    # Notice that K is a bit different than in other articles because each segment is now its own Taylor series, which has the consequence of larger error
    K = np.ceil( -1  + 2* np.log(2*r/epsilon_HS)/np.log(np.log(2*r/epsilon_HS))  ) # We 
    delta = eps_HS / t # Alternatively we can substitute t by r changing delta in the following line to 1/2. t represents L in the main text (see before eq 21 in the original article)
    tau = 1/np.ceil(2*lambd_T) # tau = t/ np.ceil(2 * lambd_T * t)
    M = np.max(16*tau/delta * (2*lambd_U_V + lambd_T), K**2)
    
    rot_FFFT = 2*N/2*np.log2(N)
    rot_U = 4*K
    rot_COEF = 2**(np.ceil(np.log2(K) + 1))
    rot_prep = 16*N
    
    epsilon_SS = 1e-2
    consistent = False
    while not consistent:
        rot_exp_U_V = rot_FFFT + N*np.log2(1/epsilon_SS) + N
        num_rotations = ((((2*rot_prep + 2* rot_FFFT + 2*np.log(M)*rot_exp_U_V)*K * rot_U) + 2*rot_COEF)*3 + rot_exp_U_V)*r
        proposed_eps_SS = eps_S / num_rotations
        if proposed_eps_SS < epsilon_SS:
            consistent = True
        else:
            epsilon_SS /= 10
            
    # Cost
    exp_U_V= 46*N*(np.log(1/eps_SS))**2+8*N + 8*N*np.log2(1/eps_SS)*np.log2(N) + 4*N*np.log(N)
    COEF = rot_COEF * (10 + 12*np.log2(K))
    U = 8*(np.log2(M) + np.log2(1/eps_SS))
    ADD = 4*np.log2(K)
    Comp = 8*np.log2(M)
    FFFT = (2 + 4*np.log(1/eps_SS))*N*np.log2(N) - 4*np.log2(1/eps_SS)*N
    Prep = 2**9*(1 + np.log2(N))+2**6*3*N*np.log2(1/eps_SS)
    Select = 8*N
    REF = 16*(np.log2(Gamma) + 2*np.log(K+1)+ 2*np.log(M))
    
    cost = ((((2*Prep + Select + 2*FFFT + 2*np.log(M)*exp_U_V)*K + (3*K+1)*ADD + K*Comp + 4*K*U +2*COEF)*3 + 2*REF) + exp_U_V)*r
    
    return cost

## Sublinear scaling and interaction picture babbush2019quantum
def sublinear_scaling_interaction(N, eta, Gamma, lambd_T, lambd_U_V, eps_S, eps_HS, eps_PEA, eps_mu, eps_M_0):
    ''' 
    See the interaction_picture function for more background
    J represents the number of atoms
    In this article there are three additional sources of error
        - the precision on preparing the amplitudes sqrt(zeta_l), eps_mu
        - the precision on the position of the atomic nuclei, 1/delta_R. In the article we take log(1/delta_R) < 1/3 log(N)
        - The precision due to the finite value of M_0 = eta N t / eps_M_0
        
    The algorithm follows a very similar structure to that of the interaction_picture one.
    '''
    
    ### IMPORTANT: SHOULD WE ALSO MEASURE THE COST OF ANTISYMMETRIZATION OF THE INITIAL STATE?: MAKES SENSE TO ME https://www.nature.com/articles/s41534-018-0071-5
    
    t = 4.7/eps_PEA
    r = lambd_U_V*t # lambd_T is necessary to take tau = 1
    
    # Notice that K is a bit different than in other articles because each segment is now its own Taylor series, which has the consequence of larger error
    K = np.ceil( -1  + 2* np.log(2*r/epsilon_HS)/np.log(np.log(2*r/epsilon_HS))  ) # We 
    delta = eps_HS / t # Alternatively we can substitute t by r changing delta in the following line to 1/2. t represents L in the main text (see before eq 21 in the original article)
    tau = 1/np.ceil(2*lambd_U_V) # tau = t/ np.ceil(2 * lambd_T * t)
    M = np.max(16*tau/delta * (lambd_U_V + 2*lambd_T), K**2)
    M0 = eta * N * tau / (eps_M_0/r)
    
    rot_exp_T = np.log2(eta) + 2*np.log2(N)
    rot_select_1 = 1/3*np.log2(N) + 2
    rot_Subprepare  = 2 # Only the two rotations from Uniform in Subprepare
    rot_COEF = 2**(np.ceil(np.log2(K) + 1))
    
    num_rotations = (((2*np.log(M)*rot_exp_T + rot_select_1)*K + 2*rot_COEF)*3 + rot_exp_T )*r
    eps_SS = eps_S / num_rotations
    
    num_Subprepare = 2*3*K*3*r
    eps_mus = eps_mu / num_Subp
    
    Subprep = 4*J + 4*np.log(1/eps_mus) +8*np.log2(1/eps_SS)+  12*np.log2(J)
    n = 1/3*np.log2(N) + 1
    Prep  = 3*(79*n**2 +43*n*np.log2(M0) + 44*n)
    exp_T = rot_exp_T * 4*np.log(1/eps_SS)
    select_0 = 16*eta*np.log2(N)
    select_1 = 8*eta*np.log2(N) + 14*(np.log2(N))**2 + 4*np.log2(N)*np.log(1/eps_SS)
    
    HAM_T = 2*np.log(M)*exp_T + 2*(3*(Subprep + Prep)) + select_0 + select_1 #The 3 multiplying Subprep and Prep comes from oblivious AA
    U = 8*(np.log2(M) + np.log2(1/eps_SS))
    ADD = 4*np.log2(K)
    Comp = 8*np.log2(M)
    
    COEF = rot_COEF * (10 + 12*np.log2(K))
    REF = 16*(np.log2(Gamma) + 2*np.log(K+1)+ 2*np.log(M))
    
    cost = (((4*K*U + K*Comp + (3*K + 1)*ADD + K*HAM_T) + 2*COEF)*3  + 2*REF)*r
    
    antisymmetrization = 3*eta*np.log2(eta)*(np.log2(eta)-1)*(2* np.ceil(np.log2(eta**2)) + np.log(N))
    
    return cost + antisymmetrization

# Finding the molecule parameters
# Docs https://quantumai.google/reference/python/openfermion/

from openfermion.chem import geometry_from_pubchem, MolecularData
from openfermionpsi4 import run_psi4
from  openfermion.transforms  import  get_fermion_operator,  jordan_wigner
import openfermion
from openfermion.utils import Grid
from openfermion.hamiltonians import plane_wave_external_potential, plane_wave_potential, plane_wave_kinetic
from openfermion.hamiltonians import plane_wave_hamiltonian
from openfermion.hamiltonians import dual_basis_external_potential, dual_basis_potential, dual_basis_kinetic

methane_geometry = geometry_from_pubchem('methane')
print(methane_geometry)

basis = 'sto-3g'

molecule = MolecularData(methane_geometry, basis, multiplicity = 1)
print(molecule)

molecule = run_psi4(molecule,run_scf=True,
                        run_mp2=True,
                        #run_cisd=False,
                        #run_ccsd=True,
                        run_fci=False
                   )

'''
To obtain these Hamiltonians one must choose to study the system without a spin degree of freedom (spinless),
one must the specify dimension in which the calculation is performed (n_dimensions, usually 3),
one must specify how many plane waves are in each dimension (grid_length)
and one must specify the length scale of the plane wave harmonics in each dimension (length_scale)
and also the locations and charges of the nuclei.

Taken from https://quantumai.google/openfermion/tutorials/intro_to_openfermion
'''

grid = Grid(dimensions = 3, length = 5, scale = 1.) # La complejidad
plane_wave_H = plane_wave_hamiltonian(grid, methane_geometry, True)

fermionic_hamiltonian = get_fermion_operator(molecule.get_molecular_hamiltonian())
plane_waves_hamiltonian = openfermion.get_diagonal_coulomb_hamiltonian(fermionic_hamiltonian)
plane_waves_hamiltonian