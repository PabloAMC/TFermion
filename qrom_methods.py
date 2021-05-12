import numpy as np

class QROM_methods:

    def __init__(self):
        pass

    ## Linear T complexity (babbush2018encoding)
    def linear_T(self, N, lambd, epsilon_PEA, epsilon_S, Ham_norm):
        '''To be used in plane wave basis'''
        t = 4.7/epsilon_PEA
        r = lambd*t
        
        mu = np.ceil(np.log(2*np.sqrt(2)*lambd/epsilon_PEA) + np.log(1 + epsilon_PEA/(8*lambd)) + np.log(1 - (Ham_norm/lambd)**2))
        
        D = 3 #dimension of the model
        M = (N/2)**3

        # The number of total rotations is r*2* number of rotations for each preparation P (in this case 2D+1)
        epsilon_SS = epsilon_S / (r*2*(2*D+1))
        z_rot_synt = self.tools.z_rotation_synthesis(epsilon_SS) #todo: see table 4 log 1/eps_ss + 10

        def uniform_cost(L, k=0, z_rot_synt = z_rot_synt, controlled = False):
            if controlled:
                return 2*k+10*np.log2(L) + 2*z_rot_synt
            else:
                return 8*np.log2(L) + 2*z_rot_synt

        def QROM_cost(N): return 4*N

        compare = self.tools.compare_cost(mu)
        sum = self.tools.sum_cost(D*np.log2(M))
        Fredkin_cost = 4 # The controlled swaps = 1 Toffoli

        Subprepare = QROM_cost(3*M**D) + uniform_cost(3) + D*uniform_cost(M) + 2*compare + (3+D*np.log2(M))*Fredkin_cost
        Prepare = Subprepare + D*uniform_cost(M, controlled=True) + D*np.log2(M)*Fredkin_cost + sum + 2*self.tools.multi_controlled_not(np.log2(N))
        
        Select = 3*QROM_cost(N) + 2*np.log2(N)*Fredkin_cost
        
        Reflexion = self.tools.multi_controlled_not(2*np.log2(N)+2*mu + N)
        return r*(2*Prepare + Reflexion + Select) #todo: rotations in the quantum walk

    ## Sparsity and low rank factorization (berry2019qubitization)
    def sparsity_low_rank(self, N, lambd, epsilon_PEA, epsilon_S, L, Ham_norm, sparsity_d = None):

        t = 4.7/epsilon_PEA
        r = lambd*t
        
        mu = np.ceil(np.log(2*np.sqrt(2)*lambd/epsilon_PEA) + np.log(1 + epsilon_PEA/(8*lambd)) + np.log(1 - (Ham_norm/lambd)**2)) #todo:in all mu definitions, the last Ham_norm/lambd to be substituted by max(Ham_norm/lambd, c) with c = .75 or so

        # Rotations are used in the Uniform protocol as well as in the ancilla to decrease the Success amplitude
        epsilon_SS = epsilon_S/ (r*2*(2*(12 +1)+6)) #first 2 is Prepare and Prepare^+, second Prepare is for the two rotations in each Uniform. Finally we have Uniform_{N/2}, Uniform_L and the ancilla rotations to decrease success prob.
        z_rot_synt = self.tools.z_rotation_synthesis(epsilon_SS) #todo: see table 4 log 1/eps_ss + 10
        rot_synt = self.tools.rotation_synthesis(epsilon_SS) #todo: see table 12 log 1/eps_ss + 10

        def uniform_cost(L, k=0, z_rot_synt = z_rot_synt, controlled = False):
            if controlled:
                return 2*k+10*np.log2(L) + 2*z_rot_synt
            else:
                return 8*np.log2(L) + 2*z_rot_synt

        def QROM_cost(N): return 4*N # To be used only in Select. In prepare we use the QROAM
        
        def closest_power(x):
            possible_results = np.floor(np.log2(x)), np.ceil(np.log2(x))
            return min(possible_results, key= lambda z: abs(x-2**z))

        Amplitude_amplification = 2*3*2*uniform_cost(N/2) + uniform_cost(L) + 2*3*rot_synt + 2*2*self.tools.multi_controlled_not(np.log2(N))

        if sparsity_d is not None:
            d = sparsity_d
        else:
            d = L(N**2/8 + N/4)
        M = np.log(N**2) + mu
        kc = 2**closest_power(np.sqrt(d/M))
        ku = 2**closest_power(np.sqrt(d))
        QROAM = 4*(np.ceil(d/kc)+4*M*(kc-1)+2*np.ceil(d/ku) + 4*ku)
        
        compare = self.tools.compare_cost(mu)
        Fredkin_cost = 4 # The controlled swaps = 1 Toffoli
        
        sum = self.tools.sum_cost(np.ceil(np.log2(N/2)))
        mult = self.tools.multiplication_cost(np.ceil(np.log2(N/2)))
        continuous_register = 2*mult + 3*sum

        # In the same order as depicted in figure 11 in PHYS. REV. X 8, 041015
        Prepare =  Amplitude_amplification + continuous_register + QROAM + compare + np.ceil(np.log2(L*N**2/4))*Fredkin_cost

        Select = 2*(2*QROM_cost(N) + 2*2*self.tools.multi_controlled_not(np.log2(N))) # The initial 2 is due to Select_1 and Select_2. See figure 1 in original article.

        Reflexion = self.tools.multi_controlled_not(2*np.log2(N)+2*mu + N)
        return r*(2*Prepare+ Reflexion + Select)