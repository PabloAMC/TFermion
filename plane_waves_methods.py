import numpy as np
import sympy
import math

from scipy import integrate

class Plane_waves_methods:

    def __init__(self, tools):
        self.tools = tools

    # Low depth quantum simulation of materials (babbush2018low) Trotter
    def low_depth_trotter(self, epsilons, N, eta, Omega):

        epsilon_PEA = epsilons[0]
        epsilon_HS = epsilons[1]
        epsilon_S = epsilons[2]
        
        t = 4.7/epsilon_PEA
        sum_1_nu = 4*np.pi*(np.sqrt(3)*N**(1/3)/2 - 1) + 3 - 3/N**(1/3) + 3*self.tools.I(N**(1/3))
        max_V = eta**2/(2*np.pi*Omega**(1/3))*sum_1_nu
        max_U = eta**2/(np.pi*Omega**(1/3))*sum_1_nu
        nu_max = np.sqrt(3*(N**(1/3))**2)
        max_T = 2*np.pi**2*eta/(Omega**(2/3))* nu_max**2
        
        r = np.sqrt(2*t**3/epsilon_HS *(max_T**2*(max_U + max_V) + max_T*(max_U + max_V)**2))

        # Arbitrary precision rotations, does not include the Ry gates in F_2
        single_qubit_rotations = 8*N + 8*N*(8*N-1) + 8*N + N*np.log2(N/2) # U, V, T and FFFT single rotations
        epsilon_SS = epsilon_S/single_qubit_rotations
        
        exp_UV_cost = (8*N*(8*N-1) + 8*N)*self.tools.pauli_rotation_synthesis(epsilon_SS)
        exp_T_cost = 8*N*self.tools.pauli_rotation_synthesis(epsilon_SS)
        F2 = 2
        FFFT_cost = N/2*np.log2(N)*F2 + N/2*(np.log2(N)-1)*self.tools.pauli_rotation_synthesis(epsilon_SS) 
        
        return r*(2*exp_UV_cost + exp_T_cost + 2*FFFT_cost )

    # Low depth quantum simulation of materials (babbush2018low) Taylor
    def low_depth_taylor(self, epsilons, N, lambda_value, Lambda_value, H_norm_lambda_ratio):

        epsilon_PEA = epsilons[0]
        epsilon_HS = epsilons[1]
        epsilon_S = epsilons[2]

        '''To be used in plane wave basis'''

        D = 3 #dimension of the model
        M = (N/2)**(1/D) # Same as in linear-T method. See also beginning of appendix I in https://journals.aps.org/prx/pdf/10.1103/PhysRevX.8.011044.

        t = 4.7/epsilon_PEA
        r = t*Lambda_value/np.log(2)

        K = np.ceil( -1  + 2* np.log(2*r/epsilon_HS)/np.log(np.log(2*r/epsilon_HS)+1)) 

        epsilon_SS = epsilon_S /(r*3*2*K*(2+4*D+2)) # In the sum the first 2 is due to Uniform_3, next 2D are due to 2 uses of Uniform_M^{otimes D}, and the final two due to the controlled rotation theta angles
        
        mu = np.ceil(np.log2(2*np.sqrt(2)*Lambda_value/epsilon_PEA) + np.log2(1 + epsilon_PEA/(8*lambda_value)) + np.log2(1 - (H_norm_lambda_ratio)**2))
        
        # The number of total rotations is r*2* number of rotations for each preparation P (in this case 2D+1)
        z_rot_synt = self.tools.pauli_rotation_synthesis(epsilon_SS)

        def uniform_cost(L, k=0, z_rot_synt = z_rot_synt, controlled = False):
            if controlled:
                return 2*k+10*np.log2(L) + 2*z_rot_synt
            else:
                return 8*np.log2(L) + 2*z_rot_synt

        def QROM_cost(N): return 4*N

        compare = self.tools.compare_cost(mu)
        sum = self.tools.compare_cost(D*np.log2(M))
        Fredkin_cost = 4 # The controlled swaps

        Subprepare = QROM_cost(3*M**D) + uniform_cost(3) + D*uniform_cost(M) + 2*compare + (3+D*np.log2(M))*Fredkin_cost
        Prepare = Subprepare + D*uniform_cost(M, controlled=True) + D*np.log2(M)*Fredkin_cost + sum + 2*self.tools.multi_controlled_not(np.log2(N))
        
        Select = 3*QROM_cost(N) + 2*np.log2(N)*Fredkin_cost
        crot_synt = self.tools.c_pauli_rotation_synthesis(epsilon_SS) # due to the preparation of the theta angles
        prepare_beta = K*(Prepare + crot_synt) # The 2*rot_synt is due to the preparation of the theta angles
        QPE_adaptation = self.tools.multi_controlled_not(np.ceil(K/2) + 1) 
        select_V = K*(Select) + QPE_adaptation

        R = self.tools.multi_controlled_not(2*np.log2(N)+2*mu+N) # Based on the number qubits needed in the Linear T QRom article
        result = r*(3*(2*prepare_beta + select_V) + 2*R)

        return result

    # Low depth quantum simulation of materials (babbush2018low) On-the fly
    def low_depth_taylor_on_the_fly(self, epsilons, N, eta, Gamma, lambda_value, Omega, J, x_max):
        
        epsilon_PEA = epsilons[0]
        epsilon_HS = epsilons[1]
        epsilon_S = epsilons[2]
        epsilon_H = epsilons[3]
        eps_tay = epsilons[4]
        
        '''To be used in plane wave basis
        J: Number of atoms
        '''
        sum_1_nu = 4*np.pi*(np.sqrt(3)*N**(1/3)/2 - 1) + 3 - 3/N**(1/3) + 3*self.tools.I(N**(1/3))
        sum_nu = self.quadratic_sum(int(N**(1/3)))
        lambda_value = ((2*eta+1)/(4*np.pi*Omega**(1/3)) - (np.pi**2)/(N*Omega**(2/3)))*(8*N)**3 
        t = 4.7/epsilon_PEA
        r = t*lambda_value/np.log(2)
        
        K = np.ceil( -1  + 2* np.log(2*r/epsilon_HS)/np.log(np.log(2*r/epsilon_HS)+1))
        zeta = epsilon_H/(r*Gamma*3*2*K)
        max_W = (2*eta+1)/(8*Omega**(1/3)*np.pi)- (np.pi**2)/(2*N*Omega**(2/3))
        M = max_W/zeta #the alternative was something like  M = Lambda_value*Gamma*r*3*2*K/epsilon_H with Lambda = max_W

        # x_max = max value of one dimension
        x = sympy.Symbol('x')
        epsilon_SS = epsilon_S / (2*K*2*3*r) # Due to the theta angles c-rotation in prepare_beta

        number_taylor_series = r* 3* 2*2*K*(J+1)
        eps_tay_s = eps_tay / number_taylor_series
        cos_order = self.tools.order_find(lambda x: math.cos(x), function_name = 'cos', e = eps_tay_s, xeval = x_max)

        n = np.ceil(np.ceil(np.log2(M))/3) #each coordinate is a third

        sum = self.tools.sum_cost(n)
        mult = self.tools.multiplication_cost(n)
        div = self.tools.divide_cost(n)

        angle_to_0_2pi_interval = div + mult + sum # angle <- angle - lceil angle/2pi rceil * 2pi
        cordic = 2*cos_order + mult 

        prepare_p_equal_q = (3*mult) + (3*sum) + (3*mult+2*sum)+((3*mult+2*sum) + (cordic+angle_to_0_2pi_interval))*J + (mult+div + J*(mult+div)) 
        prepare_p_neq_q = (3*mult) + (3*mult+2*sum) + cordic+angle_to_0_2pi_interval + div +mult
        prepare_p_q_0 = 2*mult

        sample_w = prepare_p_equal_q + prepare_p_neq_q + prepare_p_q_0

        kickback = 2*(self.tools.sum_cost(np.log2(M)) + self.tools.compare_cost(np.log2(M))) # The 2 accounts for unpreparing the ancilla

        prepare_W = 2*sample_w + kickback
        crot_synt = self.tools.c_pauli_rotation_synthesis(epsilon_SS)
        prepare_beta = K*(prepare_W + crot_synt)
        select_H = (12*N + 8*np.log2(N))
        QPE_adaptation = self.tools.multi_controlled_not(np.ceil(K/2) + 1) 
        select_V = K*select_H + QPE_adaptation

        R = self.tools.multi_controlled_not((K+1)*np.log2(Gamma) + N) # The prepare qubits and the select qubits (in Jordan-Wigner there are N)
        result = r*(3*(2*prepare_beta + select_V) + 2*R)
        
        return result

    def f(self, x, y):
            return 1/(x**2 + y**2)
    def I(self, N0):
        return integrate.nquad(self.tools.f, [[1, N0],[1, N0]])[0]

    def quadratic_sum(self, N): return N*(N+1)*(2*N + 1)**3