import numpy as np

class QROM_methods:

    def __init__(self):
        pass

    ## Linear T complexity (babbush2018encoding)
    def linear_T(self, N, lambd, eps_PEA, eps_SS):
        '''To be used in plane wave basis'''
        t = 4.7/eps_PEA
        r = lambd*t
        
        mu = np.ceil(np.log(2*np.sqrt(2)*lambd/eps_PEA) + np.log(1 + eps_PEA/(8*lambd)) + np.log(1 - (Ham_norm/lambd)**2))
        
        eps_SS = eps_S / (r*2*P)
        
        S = 12*N+8*np.log(N)
        P = 6*N + 40*np.log(N)+ 24*np.log(1/eps_SS) + 10*mu
        
        return r*(2*P + S)

    ## Sparsity and low rank factorization (berry2019qubitization)
    def sparsity_low_rank(self, N, lambd, eps_PEA, eps_SS, L):
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