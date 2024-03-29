import numpy as np

class QROM_methods:

    def __init__(self, tools):
        self.tools = tools

    ## Linear T complexity (babbush2018encoding)
    def linear_t(self, epsilons, p_fail, N, lambda_value, H_norm_lambda_ratio):

        epsilon_QPE = epsilons[0]
        epsilon_S = epsilons[1]

        '''To be used in plane wave basis'''
        t = np.pi/(2*epsilon_QPE)*(1/2+1/(2*p_fail))
        r = np.ceil(lambda_value*t)
        
        mu = np.ceil(np.log2(2*np.sqrt(2)*lambda_value/epsilon_QPE) + np.log2(1 + epsilon_QPE/(8*lambda_value)) + np.log2(1 - (H_norm_lambda_ratio)**2))
        
        D = 3 #dimension of the model
        M = (N/2)**(1/D) # eq 45 in the original article

        # The number of total rotations is r*2* number of rotations for each preparation P (in this case 2D+1)
        epsilon_SS = epsilon_S / (r*2*(2*D+1))
        z_rot_synt = self.tools.pauli_rotation_synthesis(epsilon_SS)

        def uniform_cost(L, k=0, z_rot_synt = z_rot_synt, controlled = False):
            if controlled:
                return 2*k+10*np.ceil(np.log2(L)) + 2*z_rot_synt
            else:
                return 8*np.ceil(np.log2(L)) + 2*z_rot_synt

        def QROM_cost(N): return 4*N

        compare = self.tools.compare_cost(mu)
        sum = self.tools.sum_cost(D*np.ceil(np.log2(M)))
        Fredkin_cost = 4 # The controlled swaps = 1 Toffoli

        Subprepare = QROM_cost(3*M**D) + uniform_cost(3) + D*uniform_cost(M) + 2*compare + (3+D*np.ceil(np.log2(M)))*Fredkin_cost # Fig 15 in the original article
        Prepare = Subprepare + D*uniform_cost(M, controlled=True) + D*np.ceil(np.log2(M))*Fredkin_cost + sum + 2*self.tools.multi_controlled_not(np.ceil(np.log2(N))) # Fig 16 in the original article
        
        Select = 3*QROM_cost(N) + 2*np.ceil(np.log2(N))*Fredkin_cost # Fig 14 in the original paper
        
        Reflexion = self.tools.multi_controlled_not(2*np.ceil(np.log2(N))+2*mu + N)
        return r*(2*Prepare + Reflexion + Select)

    ## Sparsity and low rank factorization (berry2019qubitization)
    def sparsity_low_rank(self, epsilons, p_fail, N, lambda_value, L, H_norm_lambda_ratio, sparsity_d = None):

        epsilon_QPE = epsilons[0]
        epsilon_S = epsilons[1]

        t = np.pi/(2*epsilon_QPE)*(1/2+1/(2*p_fail))
        r = np.ceil(lambda_value*t)
        
        mu = np.ceil(np.log2(2*np.sqrt(2)*lambda_value/epsilon_QPE) + np.log2(1 + epsilon_QPE/(8*lambda_value)) + np.log2(1 - (H_norm_lambda_ratio)**2))

        # Rotations are used in the Uniform protocol as well as in the ancilla to decrease the Success amplitude
        epsilon_SS = epsilon_S/ (r*2*(2*(12 +1)+6)) #first 2 is Prepare and Prepare^+, second Prepare is for the two rotations in each Uniform. Finally we have Uniform_{N/2}, Uniform_L and the ancilla rotations to decrease success prob.
        z_rot_synt = self.tools.pauli_rotation_synthesis(epsilon_SS)
        rot_synt = self.tools.pauli_rotation_synthesis(epsilon_SS)

        compare = self.tools.compare_cost(np.ceil(np.log2(N/2)))

        def uniform_cost(L, k=0, z_rot_synt = z_rot_synt, controlled = False):
            if controlled:
                return 2*k+10*np.ceil(np.log2(L)) + 2*z_rot_synt
            else:
                return 8*np.ceil(np.log2(L)) + 2*z_rot_synt

        def QROM_cost(N): return 4*N # To be used only in Select. In prepare we use the QROAM
        
        def closest_power(x):
            possible_results = np.floor(np.log2(x)), np.ceil(np.log2(x))
            return min(possible_results, key= lambda z: abs(x-2**z))

        # In front of uniform_cost(N/2), there is a multiplier of 3 due to Amplitude amplification, and a multiple of 4 due to p, q, r, s
        Amplitude_amplification = 2*3*(2*uniform_cost(N/2)) + 2*3*rot_synt+ 2*3*compare + 2*self.tools.multi_controlled_not(np.ceil(np.log2(N)))

        if sparsity_d is not None:
            d = sparsity_d
        else:
            d = (2*L+1)*(N**2/8 + N/4)
        M = np.ceil(np.log2(N**2)) + mu
        kc = 2**closest_power(np.sqrt(d/M))
        ku = 2**closest_power(np.sqrt(d))
        QROAM = 4*(np.ceil(d/kc)+M*(kc-1)+np.ceil(d/ku) + ku) # Includes the cost in Prepare and Prepare^\dagger

        compare = self.tools.compare_cost(mu)
        Fredkin_cost = 4 # The controlled swaps = 1 Toffoli
        controlled_swap_L = np.ceil(np.log2(L))*Fredkin_cost

        Step_1_state_preparation = uniform_cost(L) + QROM_cost(L) + compare + controlled_swap_L
        Step_2_state_preparation = self.tools.multi_controlled_not(np.ceil(np.log2(L)))

        controlled_swap_p_q = np.ceil(np.log2(N**2/4))*Fredkin_cost
        
        sum = self.tools.sum_cost(np.ceil(np.log2(N/2)))
        mult = self.tools.multiplication_cost(np.ceil(np.log2(N/2)))
        continuous_register = 2*mult + 3*sum

        # In the same order as depicted in figure 11 in PHYS. REV. X 8, 041015; include the QROAM in the final cost as encompases both prepare and unprepare
        Prepare = Amplitude_amplification + Step_1_state_preparation + Step_2_state_preparation + 2*continuous_register + 2*(compare + controlled_swap_p_q)

        Select = 2*(2*QROM_cost(N) + 2*2*self.tools.multi_controlled_not(np.ceil((np.log2(N))))) # The initial 2 is due to Select_1 and Select_2. See figure 1 in original article.

        Reflexion = self.tools.multi_controlled_not(2*np.ceil(np.log2(N))+2*mu + N)
        return r*(2*Prepare + QROAM+ Reflexion + Select)