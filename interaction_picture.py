import numpy as np

class Interaction_picture:

    def __init__(self, tools):
        self.tools = tools

    def interaction_picture(self, epsilons, N, Gamma, lambd_T, lambd_U_V):

        epsilon_S = epsilons[0]
        epsilon_HS = epsilons[1]
        epsilon_PEA = epsilons[2]

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
            > N log 1/epsilon_SS Phase operators
            > N Multiplications
        '''

        t = 4.7/epsilon_PEA
        r = lambd_T*t/np.log(2) # lambd_T is necessary to take tau = 1
        
        K = np.ceil( -1  + 2* np.log(2*r/epsilon_HS)/np.log(np.log(2*r/epsilon_HS)+1)) 
        M = np.max(16*t*np.log(2)/epsilon_HS * (2*lambd_U_V + lambd_T), K**2)
        
        # Now we count the number of individual rotations from each source: 
        rot_FFFT = N/2*np.log2(N/2) 
        rot_U = 8*N # controlled
        rot_COEF = self.tools.arbitrary_state_synthesis(K)
        rot_prep = self.tools.arbitrary_state_synthesis(8*N) 
        rot_PHASE = np.ceil(np.log2(8*N)) # controlled
        rot_uniform = 2

        # A prefactor of x2 indicates controlled rotation        
        num_rotations = rot_FFFT* r*(2+3*K*2) + 2*rot_U* r*(1+3*K*2) + rot_COEF* r*3*2 + rot_prep* r*3*K*2 + 2*rot_PHASE* r*(1+3*K*2) + rot_uniform* r*3*(4*(K-1)+2)

        epsilon_SS = epsilon_S / num_rotations

        # Rotations U+V
        ADD = self.tools.sum_cost(np.ceil(np.log2(8*N)))
        F2 = 2
        FFFT_cost = N/2*np.log2(N)*F2 + N/2*(np.log2(N)-1)*self.tools.pauli_rotation_synthesis(epsilon_SS)
        NORM = self.tools.multiplication_cost(np.ceil(np.log2(8*N)))
        mult = self.tools.multiplication_cost(2*np.ceil(np.log2(8*N))) # Vk multiplication

        PHASE = rot_PHASE* self.tools.c_z_rotation(epsilon_SS)
        exp_V = 2*(ADD+FFFT_cost + NORM + mult) + PHASE
        exp_U = rot_U * self.tools.c_z_rotation(epsilon_SS)
        exp_U_V = exp_V+exp_U

        # Qubitization of T
        Prepare = rot_prep*self.tools.pauli_rotation_synthesis(epsilon_SS)
        Select = 8*N
        T_qubitization = 2*FFFT_cost + 2*Prepare + Select

        # HAM-T
        HAM_T = 2*exp_U_V + T_qubitization

        #Uniform
        z_rot_synt = self.tools.pauli_rotation_synthesis(epsilon_SS)
        def uniform_cost(L, k=0, z_rot_synt = z_rot_synt, controlled = False):
            if controlled:
                return 2*k+10*np.log2(L) + 2*z_rot_synt
            else:
                return 8*np.log2(L) + 2*z_rot_synt
        Uniform = uniform_cost(np.ceil(np.log2(M)))

        # DYS_K
        Compare = self.tools.compare_cost(np.ceil(np.log2(M)))
        SWAP = 0 # It is not controlled, therefore all Clifford
        ADD_c = self.tools.sum_cost(np.ceil(1+np.log2(K+1)))
        ADD_b = self.tools.sum_cost(np.ceil(np.log2(K+1)))

        DYS_0 = 2*Uniform + 2*ADD_c + ADD_b + HAM_T
        DYS_k = 4*Uniform + 2*ADD_c + ADD_b + Compare + SWAP + HAM_T 

        DYS_K = (K-1)*DYS_k + DYS_0 + ADD_b

        # TDS
        COEF = rot_COEF*self.tools.pauli_rotation_synthesis(epsilon_SS)
        TDS_2 = 2*COEF + DYS_K
        R = self.tools.multi_controlled_not(N+np.ceil(np.log2(Gamma)) + 2 + np.ceil(np.log2(K+1)) + np.ceil(np.log2(M)))
        TDS = 2*R + 3*TDS_2

        cost = r*(exp_U_V + TDS)
        return cost

    ## Sublinear scaling and interaction picture babbush2019quantum
    def sublinear_scaling_interaction(self, epsilons, N, eta, Gamma, lambd_T, lambd_U_V, J):
        
        epsilon_S = epsilons[0]
        epsilon_HS = epsilons[1]
        epsilon_PEA = epsilons[2]
        epsilon_mu = epsilons[3]
        epsilon_M0 = epsilons[4]
        epsilon_R = epsilons[5]
        
        ''' 
        See the interaction_picture function for more background
        J represents the number of atoms
        In this article there are three additional sources of error
            - the precision on preparing the amplitudes sqrt(zeta_l), epsilon_mu
            - the precision on the position of the atomic nuclei, 1/delta_R. In the article we take log(1/delta_R) < 1/3 log(N)
            - The precision due to the finite value of M_0 = eta N t / epsilon_M_0
            
        The algorithm follows a very similar structure to that of the interaction_picture one.
        '''
        
        ### Main algorithm

        t = 4.7/epsilon_PEA
        r = lambd_U_V*t # lambd_T is necessary to take tau = 1
        
        # Notice that K is a bit different than in other articles 
        K = np.ceil( -1  + 2* np.log(2*r/epsilon_HS)/np.log(np.log(2*r/epsilon_HS)+1))  # Same as in the previous function

        first_term = 16*t*np.log(2)/epsilon_HS * (lambd_U_V + 2*lambd_T)
        second_term = K**2

        M = np.max(first_term, second_term) # Changes from the M in the previous function in T<->U+V

        # Deltas
        delta_M0 = epsilon_M0/ (3*2*K*3*r) # Number of times prep_nu is used bottom up counting
        delta_R = epsilon_R/ (3*N**(1/3) * K*3*r) # 3 Coordinates of size N^(1/3) x number of times the Phase operation in Select_U is used
        delta_mu = epsilon_mu/ ((2*(1+3) + 2)*K*3*r) # Number of times QROM is used, (1+3) in Prepare and 2 in Select_U

        # M0 and n
        M0 = 2**np.ceil(np.log2(1/delta_M0))
        n = np.ceil(1/3*np.log2(N))
        
        # Number of rotations
        rot_exp_T = np.ceil(2*np.log2(N**(1/3))) +3*eta  # Controlled. Based on the number of digits: 3 squares  (x2) +  3 eta sums
        rot_select_U = np.ceil((1/3)*np.log2(N))+np.ceil(np.log2(delta_R)) + 2 # Controlled. The length of the registers is (1/3)*np.log2(N) (each coord) + log delta_R +  2 (2 sums) 
        rot_Uniform = 2 # Those not included in Subprepare
        rot_Subprepare = 2 # Only the two rotations from Uniform in Subprepare (cube weighting and the Subprepare in Prepare)
        rot_Prepare_cube = self.tools.arbitrary_state_synthesis(n)
        rot_COEF = self.tools.arbitrary_state_synthesis(K)

        # A prefactor of x2 indicates controlled rotation
        num_rotations = 2*rot_exp_T* r*(1+3*K*2) + 2*rot_select_U* r*3*K + rot_Uniform* r*3*(4*K+2) + rot_Subprepare *r*3*K*2*1 + rot_Prepare_cube **r*3*K*2*3 + rot_COEF* r*3*2
        epsilon_SS = epsilon_S / num_rotations

        # Uniform
        z_rot_synt = self.tools.pauli_rotation_synthesis(epsilon_SS)
        def uniform_cost(L, k=0, z_rot_synt = z_rot_synt, controlled = False):
            if controlled:
                return 2*k+10*np.log2(L) + 2*z_rot_synt
            else:
                return 8*np.log2(L) + 2*z_rot_synt
        Uniform = uniform_cost(np.ceil(np.log2(M)))

        # Exp_T
        mult = self.tools.multiply_cost(np.ceil(np.log2(N**(1/3))))
        sum = self.tools.sum_cost(2*np.ceil(np.log2(N**(1/3))))
        # There is one extra multiplication for the (2pi)^2/Omega^(2/3) coefficient
        phase = rot_exp_T*self.tools.c_z_rotation(epsilon_SS) + mult
        exp_T = (3*eta)*mult + (3*eta)*sum + phase 

        # Prep_nu
        Fredkin_cost = 4 # Equivalent to a Toffoli
        def QROM_cost(N): return 4*N
        compare = self.tools.compare_cost(np.ceil(np.log2(1/delta_mu)))
        
        Prepare_cube = rot_Prepare_cube*self.tools.pauli_rotation_synthesis(epsilon_SS)
        Other_cube = n*self.tools.multi_controlled_not(np.ceil(np.log2(n)))

        Negative0 = self.tools.multi_controlled_not(n)
        In_the_box = n*self.tools.multi_controlled_not(4)
        Inequality = 3*mult + 2*sum + mult + 2*n*self.tools.multi_controlled_not(2*np.ceil(1/3*np.log2(N)) + np.log2(M0) + 3)

        prep_nu = (Prepare_cube + Other_cube) + Negative0 + In_the_box + Inequality

        # Prepare
        Momentum_state = 3*prep_nu + 2*self.tools.multi_controlled_not(1+n+1) # The Amplitude Amplification step: Rotations on the flag qubits indicating failure 1+n+1 for steps 2, 3 and 4
        Subprepare_J = QROM_cost(J) + uniform_cost(J) + compare + (np.ceil(np.log2(J)))*Fredkin_cost
        U_V_weighting = self.tools.pauli_rotation_synthesis(epsilon_SS)
        Prepare = U_V_weighting + Subprepare_J + Momentum_state

        # Select
        sum = self.tools.sum_cost(np.ceil(np.log2(N**(1/3))))
        c_vec_sum = 3*2*sum + 3*4*np.ceil(np.log2(N**(1/3))) # 3 components, (subs + add), add 1
        equality = self.tools.multi_controlled_not(np.ceil(np.log2(N**(1/3))))
        Select_V = 2*eta*(c_vec_sum + 2*equality)

        sum = self.tools.sum_cost(np.ceil(2*np.log2(N**(1/3))))
        max_digits = np.ceil(np.max(2*np.log2(N**(1/3)),1/delta_R))
        mult = self.tools.multiplication_cost(max_digits)
        dot_prod = 3*mult + 2*sum + mult
        phase = rot_select_U*self.tools.c_z_rotation(epsilon_SS)
        Select_U = 2*QROM_cost(J) + 2*dot_prod + phase  + eta*(c_vec_sum + 2*equality)

        CRz_on_x = 3*4 # 3 Toffolis are enough

        Select = Select_U + Select_V + CRz_on_x

        # (U+V) qubitization
        U_V_qubitization = 2*Prepare + Select

        # HAM-T
        HAM_T = 2*exp_T + U_V_qubitization

        # DYS_K
        Compare = self.tools.compare_cost(np.ceil(np.log2(M)))
        SWAP = 0 # It is not controlled, therefore all Clifford
        ADD_c = self.tools.sum_cost(np.ceil(1+np.log2(K+1)))
        ADD_b = self.tools.sum_cost(np.ceil(np.log2(K+1)))

        DYS_0 = 2*Uniform + 2*ADD_c + ADD_b + HAM_T
        DYS_k = 4*Uniform + 2*ADD_c + ADD_b + Compare + SWAP + HAM_T 

        DYS_K = (K-1)*DYS_k + DYS_0 + ADD_b

        # TDS
        COEF = rot_COEF*self.tools.pauli_rotation_synthesis(epsilon_SS)
        TDS_2 = 2*COEF + DYS_K
        R = self.tools.multi_controlled_not(N+np.ceil(np.log2(Gamma)) + 2 + np.ceil(np.log2(K+1)) + np.ceil(np.log2(M)))
        TDS = 2*R + 3*TDS_2

        cost = r*(exp_T + TDS)
        return cost