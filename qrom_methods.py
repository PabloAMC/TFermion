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

        compare = self.tools.compare(mu)
        sum = self.tools.compare(D*np.log2(M))
        Fredkin_cost = 4 # The controlled swaps = 1 Toffoli

        Subprepare = QROM_cost(3*M**D) + uniform_cost(3) + D*uniform_cost(M) + 2*compare + (3+D*np.log2(M))*Fredkin_cost
        Prepare = Subprepare + D*uniform_cost(M, controlled=True) + D*np.log2(M)*Fredkin_cost + sum + 2*self.tools.multi_controlled_not(np.log2(N))
        
        Select = 3*QROM_cost(N) + 2*np.log2(N)*Fredkin_cost
        
        return r*(2*Prepare + Select)

    ## Sparsity and low rank factorization (berry2019qubitization)
    def sparsity_low_rank(self, N, lambd, epsilon_PEA, epsilon_S, L, Ham_norm):
        t = 4.7/epsilon_PEA
        r = lambd*t
        
        mu = np.ceil(np.log(2*np.sqrt(2)*lambd/epsilon_PEA) + np.log(1 + epsilon_PEA/(8*lambd)) + np.log(1 - (Ham_norm/lambd)**2))
        d = L(N**2/8 + N/4)
        M = np.log(N**2) + mu
        
        def closest_power(x):
            possible_results = np.floor(np.log2(x)), np.ceil(np.log2(x))
            return min(possible_results, key= lambda z: abs(x-2**z))
        
        kc = 2**closest_power(np.sqrt(d/M))
        ku = 2**closest_power(np.sqrt(d))
        
        QROAM = 4*(np.ceil(d/kc)+4*M*(kc-1)+2*np.ceil(d/ku) + 4*ku)
        
        Select = (4*N + 4*np.log(N))*4 # The *4 because Toffoli -> T-gates
        
        # 7 times per prepare, we have to use Uniform
        epsilon_SS = epsilon_S/ (7*2*r)
        Uniform = 8*np.log(L) + 56*np.log(1/epsilon_SS) + 52*np.log(N/2) ### Warning, this is in T gates already!!!!
        
        Other_subprepare = mu + np.log(L) + 6*np.log(N/2)
        
        continuous_register = 2*(np.log(N/2))**2 + 3*np.log(N/2)
        
        Prepare = 4*(QROAM + Other_subprepare + continuous_register) + Uniform # The 4 is Toffoli -> T-gates
        
        return r*(2*Prepare + Select)