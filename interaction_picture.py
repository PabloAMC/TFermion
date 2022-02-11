import numpy as np
import sympy
import itertools

class Interaction_picture:

    def __init__(self, tools):
        self.tools = tools
        self.weight_T_cost = self.tools.config_variables['weight_T_cost']
        self.weight_toffoli_cost = self.tools.config_variables['weight_toffoli_cost']

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

        t = np.pi/epsilon_PEA
        r = np.ceil(lambd_T*t/(2*np.log(2))) # lambd_T is necessary to take tau = 1
        
        K = np.ceil( -1  + 2* np.log(2*r/epsilon_HS)/np.log(np.log(2*r/epsilon_HS)+1)) 
        M = np.max([16*t*np.log(2)/epsilon_HS * (2*lambd_U_V + lambd_T), K**2])
        
        # Now we count the number of individual rotations from each source: 
        rot_FFFT = N/2*np.log2(N/2) 
        rot_U = 8*N # controlled
        rot_COEF = self.tools.arbitrary_state_synthesis(np.ceil(np.log2(K)))
        rot_prep = self.tools.arbitrary_state_synthesis(np.ceil(np.log2(8*N))) 
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

        PHASE = rot_PHASE* self.tools.c_pauli_rotation_synthesis(epsilon_SS)
        exp_V = 2*(ADD+FFFT_cost + NORM + mult) + PHASE
        exp_U = rot_U * self.tools.c_pauli_rotation_synthesis(epsilon_SS)
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
                return 2*k+10*np.ceil(np.log2(L)) + 2*z_rot_synt
            else:
                return 8*np.ceil(np.log2(L)) + 2*z_rot_synt
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

    def first_quantization_qubitization(self, optimized_parameters, N, N_small, eta, lambda_zeta, Omega, cost_unit, cost_module, vec_a, amplitude_amplification = True):
        '''
        Based on the qubitization method from Fault-Tolerant Quantum Simulations of Chemistry in First Quantization

        Inputs:
            epsilons: error terms
            N: number of plane waves
            eta: number of electrons
            lambda_zeta: sum of nuclear charges (if neutral, equals eta, see eq 56)
            Omega: cell volume

        return Toffoli gate cost.
        '''
        if vec_a.size != 0:
            vec_b = 1/np.array(vec_a)
            vec_b /= np.min(vec_b)
        else:
            vec_b = np.array([1,1,1])

        epsilon_PEA = optimized_parameters[0]
        epsilon_S = optimized_parameters[3]
        br = int(np.round(optimized_parameters[5]))

        # calculate parameters
        a = 3 if amplitude_amplification else 1

        n_p, n_eta, n_eta_zeta, n_M, n_R, n_T, lambda_value = self.calculate_number_bits_parameters(optimized_parameters, N, eta, lambda_zeta, Omega, amplitude_amplification)

        r = np.ceil(np.pi*lambda_value/(2*epsilon_PEA))

        epsilon_S_HF = 1e-2 # This parameter indicates the decrease in overlap with ground state due to imperfect rotations.
        epsilon_SS_HF = epsilon_S_HF / (2*eta*(N_small-eta))

        '''cost = (2*(n_T + 4*n_eta_zeta + 2*br-12) + 14*n_eta +8*br -36+a*(3*n_p**2+15*n_p-7+4*n_M*(n_p+1))
        +lambda_zeta+self.Er(lambda_zeta) + 2*(2*n_p + 2*br-7) + 12*eta*n_p+5*(n_p-1) + 2 + 24*n_p+6*n_p*n_R +18
        +n_eta_zeta +2*n_eta + 6*n_p + n_M+16)*np.ceil(np.pi*lambda_value/(2*epsilon_PEA))
        return cost'''

        def calculate_HF_cost():

            # T gate cost for HF
            T_givens = self.tools.c_pauli_rotation_synthesis(epsilon_SS_HF)# For C-Ry, lemma 5.4 Barenco
            HF_T_cost = eta*(N_small-eta)*T_givens
            
            # toffoli gate cost for HF
            Givens = 2* (4*(np.ceil(np.log2(N_small))-2)) # Our figure 4 on Givens rotations: Using Barenco lemma 7.2: MCX + uncomputation
            HF_toffoli_cost = eta*(N_small-eta)*Givens + calculate_antisymmetrization_cost()

            if cost_unit == 'T': HF_cost = HF_T_cost
            elif cost_unit == 'Toffoli': HF_cost = HF_toffoli_cost
            elif cost_unit == 'optimization': HF_cost = HF_T_cost*self.weight_T_cost + HF_toffoli_cost*self.weight_toffoli_cost
            elif cost_unit == 'detail': HF_cost = {'T':HF_T_cost, 'Toffoli':HF_toffoli_cost}

            return HF_cost

        def calculate_antisymmetrization_cost():

            # Initial state antisymmetrization
            comparison_eta = self.tools.compare_cost(np.ceil(np.log2(eta**2)))/4
            comparison_N = self.tools.compare_cost(np.ceil(np.log2(N)))/4
            swaps_eta = np.ceil(np.log2(eta**2))
            swaps_N = np.ceil(np.log2(N))
            Step_2 = eta*np.ceil(np.log2(eta))*(np.ceil(np.log2(eta))-1)/4* (comparison_eta + swaps_eta)
            Step_4 = eta*np.ceil(np.log2(eta))*(np.ceil(np.log2(eta))-1)/4* (comparison_N + swaps_N)
            antisymmetrization_cost = Step_2*2 + Step_4 #the *2 is due to expected success rate

            return antisymmetrization_cost

        def calculate_QPE_cost():

            # Section A
            ## weigthings between T and U+V, and U and V.
            weight_T_UV = n_T-3

            epsilon_SS_QPE = epsilon_S / (r*np.max([n_R+1, n_T])) # Denominator is r times size of gradient phase state: point 3 after eq C1 in https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.2.040332

            eq_superp_T_UV = 3*n_eta_zeta + 2*br - 9
            ineq_test = n_eta_zeta - 1
            weight_U_V = eq_superp_T_UV +  ineq_test + 1

            prep_qubit_TUV = weight_T_UV + weight_U_V

            ## superposition between i and j
            bin_super = 3*n_eta + 2*br - 9
            equality_test = n_eta
            inv_equality_test = n_eta
            inv_bin_super = 3*n_eta + 2*br - 9

            prep_i_j = 2*bin_super + equality_test + inv_equality_test + 2*inv_bin_super

            ## success check
            success_check = 3

            # Section B: Qubitization of T

            ## Superposition w,r,s
            sup_w = 3*2 + 2*br - 9 # = 3*n + 2*br - 9 with n = 2. br is suggested to be 8
            #T_sup_w = self.tools.pauli_rotation_synthesis(epsilon_SS_QPE) + self.tools.c_pauli_rotation_synthesis(epsilon_SS_QPE)
            sup_r = n_p - 2
            prep_wrs_T = 2*sup_r +sup_w # 2 for r and s

            ## Sel T
            control_swap_i_j_ancilla = 2*2*(eta-2) #unary iteration for i and j, and for in and out
            swap_i_j_ancilla = 2*2*3*eta*n_p # 3 components, 2 for i and j, 2 for in and out. Also used in Sel_U_V
            cswap_p_q = control_swap_i_j_ancilla + swap_i_j_ancilla

            control_copy_w = 3*(n_p-1)
            copy_r = n_p - 1
            copy_s = n_p - 1
            control_phase = 1
            erasure = 0
            control_qubit_T = 1
            Sel_T = control_copy_w + copy_r + copy_s + control_phase + erasure + control_qubit_T

            # Section C: Preparation U and V
            nested_boxes = n_p-1
            coord_prep = 3*(n_p-1)
            minus_zero_flag = 3*n_p +2
            inside_box_test = 3*n_p

            # if we are summing n terms we have to do n-1 sums; we keep the first n_p bits of the binary representation
            # we also assume that all the vec_b components are 1.something
            n_sums = sum([np.binary_repr(int(vec_b[i]*2**n_p))[:n_p+1].count('1') for i in range(len(vec_b))]) - 1

            sums = n_sums*self.tools.sum_cost(2*n_p+2)/4
            squares = 3*n_p**2-n_p-1
            multiplication = 2*n_M*(2*n_p+2)-n_M
            comparison = 2*n_p + n_M + 2
            success_flag = 3

            inversion_c_hadamards = nested_boxes + coord_prep

            Prep_1_nu_and_inv = nested_boxes + coord_prep + minus_zero_flag + inside_box_test + 2*sums + squares + multiplication + comparison + success_flag + inversion_c_hadamards

            QROM_Rl = lambda_zeta + self.Er(lambda_zeta)

            # Section D: Sel U and V

            swap_i_j_ancilla = 2*2*3*eta*n_p # 3 components, 2 for i and j, 2 for in and out  (Duplicated from Sel T)

            ## Controlled sum and substraction with change from signed integer to 2's complement
            signed2twos = 2*3*(n_p-2) # the 2 is for p and q, the 3 for their components
            addition = n_p+1
            controlled = n_p+1
            controlled_addition = 2*3*(addition + controlled)
            twos2signed = 2*3*(n_p) #Same as above, now with two extra qubits
            controlled_sum_substraction_nu = signed2twos + controlled_addition + twos2signed

            # No control-Z on |+> on Sel

            if n_R > n_p: # eq 97 # the product R_l \cdot k_\nu cancels the terms a_i
                U_phase = 3*(2*n_p*n_R-n_p*(n_p+1)-1)
            else:
                U_phase = 3*n_R*(n_R-1)

            #We could phase the T gates for each case, but instead it is faster to add the value into a phase gradient state: see https://quantum-journal.org/papers/q-2018-06-18-74/pdf/
            #U_phase_T_gates = (n_p+n_R+2)*self.tools.pauli_rotation_synthesis(epsilon_SS_QPE) # arbitrary single rotations. The 2 comes from summing the three components

            # Total cost of Prepare and unprepare
            Prep = 2*prep_qubit_TUV + prep_i_j + success_check + 2*prep_wrs_T + a*Prep_1_nu_and_inv + QROM_Rl

            # Total cost of Select
            Sel = cswap_p_q + Sel_T + controlled_sum_substraction_nu + U_phase

            # Rotation in definition of Q
            Rot = n_eta_zeta + 2*n_eta + 6*n_p + n_M + 16

            # Final toffoli cost
            QPE_toffoli_cost = r*(Prep + Sel + Rot)

            # Final T cost
            QPE_T_cost = np.max([n_R+1,n_T])*self.tools.pauli_rotation_synthesis(epsilon_SS_QPE)

            if cost_unit == 'T': QPE_cost = QPE_T_cost
            elif cost_unit == 'Toffoli': QPE_cost = QPE_toffoli_cost
            elif cost_unit == 'optimization': QPE_cost = QPE_T_cost*self.weight_T_cost + QPE_toffoli_cost*self.weight_toffoli_cost
            elif cost_unit == 'detail': QPE_cost = {'T': QPE_T_cost, 'Toffoli': QPE_toffoli_cost}

            return QPE_cost

        def qubit_cost():
            logical_qubits  = 3*eta*n_p + 12*n_p + 33 + 2*np.ceil(np.log2(eta))+ \
                    5*n_M + 3*n_p**2+ 4*n_M*n_p + np.ceil(np.log2(eta+2*lambda_zeta))+ \
                    np.max([n_R+1,n_T]) + np.max([5*n_R-4,5*n_p+1])
            return logical_qubits

        if cost_module == 'detail':
            return calculate_HF_cost(), calculate_QPE_cost(), qubit_cost()
        elif cost_module == 'optimization':
            return calculate_HF_cost()+calculate_QPE_cost()

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
        ### Initial state antisymmetrization
        comparison_eta = self.tools.compare_cost(np.ceil(np.log2(eta**2)))
        comparison_N = self.tools.compare_cost(np.ceil(np.log2(N)))
        swaps_eta = 4*np.ceil(np.log2(eta**2))
        swaps_N = 4*np.ceil(np.log2(N))
        Step_2 = eta*np.ceil(np.log2(eta))*(np.ceil(np.log2(eta))-1)/4* (comparison_eta + swaps_eta)
        Step_4 = eta*np.ceil(np.log2(eta))*(np.ceil(np.log2(eta))-1)/4* (comparison_N + swaps_N)
        antisymmetrization = Step_2*2 + Step_4

        ### Main algorithm

        t = np.pi/epsilon_PEA
        r = np.ceil(np.e*lambd_U_V*t/2) #Alternatively 2*lambd_U_V*t/np.log(2); lambd_T is necessary to take tau = 1
        
        # Notice that K is a bit different than in other articles 
        K = np.ceil( -1  + 2* np.log(2*r/epsilon_HS)/np.log(np.log(2*r/epsilon_HS)+1))  # Same as in the previous function
        M = np.max([16*t*np.log(2)/epsilon_HS * (lambd_U_V + 2*lambd_T), K**2]) # Changes from the M in the previous function in T<->U+V

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
        rot_Prepare_cube = self.tools.arbitrary_state_synthesis(np.ceil(np.log2(n)))
        rot_COEF = self.tools.arbitrary_state_synthesis(np.ceil(np.log2(K)))

        # A prefactor of x2 indicates controlled rotation
        num_rotations = 2*rot_exp_T* r*(1+3*K*2) + 2*rot_select_U* r*3*K + rot_Uniform* r*3*(4*K+2) + rot_Subprepare *r*3*K*2*1 + rot_Prepare_cube*r*3*K*2*3 + rot_COEF* r*3*2
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
        mult = self.tools.multiplication_cost(np.ceil(np.log2(N**(1/3))))
        sum = self.tools.sum_cost(2*np.ceil(np.log2(N**(1/3))))
        # There is one extra multiplication for the (2pi)^2/Omega^(2/3) coefficient
        phase = rot_exp_T*self.tools.c_pauli_rotation_synthesis(epsilon_SS) + mult
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
        max_digits = np.ceil(np.max([2*np.log2(N**(1/3)),1/delta_R]))
        mult = self.tools.multiplication_cost(max_digits)
        dot_prod = 3*mult + 2*sum + mult
        phase = rot_select_U*self.tools.c_pauli_rotation_synthesis(epsilon_SS)
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
        return cost + antisymmetrization

    def Ps(self, n, br): #eq 59 from https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.2.040332

            theta = 2*np.pi/(2**br)*np.round((2**br)/(2*np.pi)*np.arcsin(np.sqrt(2**(np.ceil(np.log2(n)))/(4*n)))) #eq 60
            braket = (1+(2-(4*n)/(2**np.ceil(np.log2(n))))*(np.sin(theta))**2)**2 + (np.sin(2*theta))**2
            return n/(2**np.ceil(np.log2(n)))*braket

    def calculate_lambdas(self, N, eta, lambda_zeta, Omega, n_p, M):

        # for lambda_nu see eq F6 in Low-depth article and D18 in T-Fermion
        lambda_nu = 4*np.pi*(np.sqrt(3)*N**(1/3)/2 - 1) + 3 - 3/N**(1/3) + 3*self.tools.I(N**(1/3))
        lambda_U = eta*lambda_zeta*lambda_nu/(np.pi*Omega**(1/3))
        lambda_V = eta*(eta-1)*lambda_nu/(2*np.pi*Omega**(1/3))
        
        # lambda_T is corrected in eq 71 of https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040332 for failure in comparison of i and j
        lambda_prime_T = 6*eta * np.pi**2 * 2**(2*n_p-2) / (Omega**(2/3)) #eq 71 https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.2.040332
        
        # lambda_U and lambda_V are corrected in eq 123 and 124 of https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040332
        # eps is taken from their eq 113. If a - b < eps, then a < b + eps = b(1+eps/b). We have to scale the norm of U and V by (1+eps/b) where b = lambda_nu
        eps = 4/M*(7*2**(n_p+1) + 9*n_p -11 -3*2**(-n_p))
        lambda_U_prime = lambda_U*(1+eps/lambda_nu)
        lambda_V_prime = lambda_V*(1+eps/lambda_nu)

        return lambda_U_prime, lambda_V_prime, lambda_prime_T

    def calculate_number_bits_parameters(self, optimized_parameters, N, eta, lambda_zeta, Omega, amplitude_amplification):

        _, epsilon_M, epsilon_R, _, epsilon_T, br = optimized_parameters

        # n_p
        n_p = int(np.ceil(np.log2(N**(1/3) + 1)))
        Peq = self.Ps(3,br)*self.Ps(eta+2*lambda_zeta,br)*(self.Ps(eta, br))**2

        # n_eta
        n_eta = np.ceil(np.log2(eta))
        
        # n_eta_zeta
        n_eta_zeta = np.ceil(np.log2(eta+2*lambda_zeta))
        
        # n_M
        n_M = np.ceil(np.log2( (2*eta)*(eta-1+2*lambda_zeta)*(7*2**(n_p+1)-9*n_p+11-3*2**(-n_p))/(epsilon_M*np.pi*Omega**(1/3))))

        # n_R
        n_R = np.ceil(np.log2( eta*lambda_zeta/(epsilon_R*Omega**(1/3))*self.tools.sum_1_over_nu(N)))

        # n_T
        M  = 2**n_M
        p_nu = 0
        n_mu = n_p+1        
        G = sympy.S.Catalan.evalf()
        #y = sympy.Symbol('y')
        #x = sympy.Symbol('x')
        #Ti = sympy.integrate(sympy.atan(y)/y, (y, 0, x)).evalf(subs={x:3-sympy.sqrt(8)})
        Ti = 0.171017553023190
        if n_p > 7 or np.isnan(n_M):
            p_nu = 1-float(1-3/8*(Ti - G +np.pi/2*np.log(1+np.sqrt(2))))
        elif n_p == 7 and n_M > 12: # Precalculated to avoid numerical delays
            p_nu = 0.23779
        elif n_p == 6 and n_M > 12: # Precalculated to avoid numerical delays
            p_nu = 0.23577
        elif n_p == 5 and n_M > 12: # Precalculated to avoid numerical delays
            p_nu = 0.23173
        else: # eq 40 in https://www.nature.com/articles/s41534-019-0199-y
            B_mus = {}
            p_nu = 0
            for j in range(2, n_p+4):
                B_mus[j] = []
            for nu in itertools.product(range(-2**(n_p), 2**(n_p)+1), repeat = 3):
                nu = np.array(nu)
                if list(nu) != [0,0,0]:
                    mu = int(np.floor(np.log2(np.max(abs(nu)))))+2
                    B_mus[mu].append(nu)
            for mu in range(2, (n_p+2)):
                for nu in B_mus[mu]:
                    p_nu += np.ceil(M*((2**(mu-2))/np.linalg.norm(nu))**2)/(M*2**(2*mu)*2**(n_p+2))

        # Lambda values. See eq 25 from https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.2.040332
        # and also eq 126 and 127
        lambda_U_prime, lambda_V_prime, lambda_prime_T = self.calculate_lambdas(N, eta, lambda_zeta, Omega, n_p, M)

        if amplitude_amplification:
            p_nu_amp = (np.sin(3*np.arcsin(np.sqrt(p_nu))))**2
            lambda_value = max(lambda_prime_T+lambda_U_prime+lambda_V_prime, (lambda_U_prime+lambda_V_prime/(1-1/eta))/p_nu_amp)/Peq
        else:
            lambda_value = max(lambda_prime_T+lambda_U_prime+lambda_V_prime, (lambda_U_prime+lambda_V_prime/(1-1/eta))/p_nu)/Peq
        n_T = np.ceil(np.log2( np.pi*lambda_value/epsilon_T ))


        return n_p, n_eta, n_eta_zeta, n_M, n_R, n_T, lambda_value

    def Er(self, x):
        logx = np.log2(x)
        fres = 2**(np.floor(logx/2)) + np.ceil(2**(-np.floor(logx/2))*x)
        cres = 2**(np.ceil(logx/2)) + np.ceil(2**(-np.ceil(logx/2))*x)
        return min(fres, cres)