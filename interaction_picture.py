import numpy as np

class Interaction_picture:

    def __init__(self):
        pass

    def interaction_picture(self, N, Gamma, lambd_T, lambd_U_V, epsilon_S, epsilon_HS, epsilon_PEA):
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
        
        # Notice that K is a bit different than in other articles 
        K = np.ceil( -1  + 2* np.log(2*r/epsilon_HS)/np.log(np.log(2*r/epsilon_HS))) 
        M = np.max(16*t*np.log(2)/epsilon_HS * (2*lambd_U_V + lambd_T), K**2)
        
        # Now we count the number of individual rotations from each source: 
        rot_FFFT = N/2*np.log2(N/2) 
        rot_U = 8*N 
        rot_COEF = self.tools.arbitrary_state_synthesis(np.ceil(np.log2(K))) 
        rot_prep = self.tools.arbitrary_state_synthesis(np.ceil(np.log2(8*N))) 
        rot_PHASE = np.log2(8*N) 
        
        num_rotations = rot_FFFT* r*(2+3*K*2) + rot_U* r*(1+3*K*2) + rot_COEF* r*3 + rot_prep* r*3*K*2 + rot_PHASE* r*(1+3*K*2)

        epsilon_SS = epsilon_S / num_rotations

        # Rotations U+V
        ADD = self.tools.sum_cost(np.ceil(np.log2(8*N)))
        F2 = 2
        FFFT_cost = N/2*np.log2(N)*F2 + N/2*(np.log2(N)-1)*self.tools.z_rotation_synthesis(epsilon_SS)
        NORM = self.tools.multiplication_cost(np.ceil(np.log2(8*N)))

        PHASE = rot_PHASE* self.tools.z_rotation_synthesis(epsilon_SS)
        exp_V = 2*(ADD+FFFT_cost + NORM) + PHASE
        exp_U = rot_U * self.tools.z_rotation_synthesis(epsilon_SS)
        exp_U_V = exp_V+exp_U

        # Qubitization of T
        Prepare = rot_prep*self.tools.rotation_synthesis(epsilon_SS)
        Select = 8*N
        T_qubitization = 2*FFFT_cost + 2*Prepare + Select

        # HAM-T
        HAM_T = 2*exp_U_V + T_qubitization

        #Uniform
        z_rot_synt = self.tools.z_rotation_synthesis(epsilon_SS)
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
        COEF = rot_COEF*self.tools.rotation_synthesis(epsilon_SS)
        TDS_2 = 2*COEF + DYS_K
        R = self.tools.multi_controlled_not(N+np.ceil(np.log2(Gamma)) + 2 + np.ceil(np.log2(K+1)) + np.ceil(np.log2(M)))
        TDS = 2*R + 3*TDS_2

        cost = r*(exp_U_V + TDS)
        return cost

    ## Sublinear scaling and interaction picture babbush2019quantum
    def sublinear_scaling_interaction(self, N, eta, Gamma, lambd_T, lambd_U_V, epsilon_S, epsilon_HS, epsilon_PEA, epsilon_mu, epsilon_M_0, J):
        ''' 
        See the interaction_picture function for more background
        J represents the number of atoms
        In this article there are three additional sources of error
            - the precision on preparing the amplitudes sqrt(zeta_l), epsilon_mu
            - the precision on the position of the atomic nuclei, 1/delta_R. In the article we take log(1/delta_R) < 1/3 log(N)
            - The precision due to the finite value of M_0 = eta N t / epsilon_M_0
            
        The algorithm follows a very similar structure to that of the interaction_picture one.
        '''
        
        ### IMPORTANT: SHOULD WE ALSO MEASURE THE COST OF ANTISYMMETRIZATION OF THE INITIAL STATE?: MAKES SENSE TO ME https://www.nature.com/articles/s41534-018-0071-5
        
        t = 4.7/epsilon_PEA
        r = lambd_U_V*t # lambd_T is necessary to take tau = 1
        
        # Notice that K is a bit different than in other articles because each segment is now its own Taylor series, which has the consequence of larger error
        K = np.ceil( -1  + 2* np.log(2*r/epsilon_HS)/np.log(np.log(2*r/epsilon_HS))) # We 
        delta = epsilon_HS / t # Alternatively we can substitute t by r changing delta in the following line to 1/2. t represents L in the main text (see before eq 21 in the original article)
        tau = 1/np.ceil(2*lambd_U_V) # tau = t/ np.ceil(2 * lambd_T * t)
        M = np.max(16*tau/delta * (lambd_U_V + 2*lambd_T), K**2)
        M0 = eta * N * tau / (epsilon_M_0/r)
        
        rot_exp_T = np.log2(eta) + 2*np.log2(N)
        rot_select_1 = 1/3*np.log2(N) + 2
        rot_Subprepare = 2 # Only the two rotations from Uniform in Subprepare
        rot_COEF = 2**(np.ceil(np.log2(K) + 1))
        
        num_rotations = (((2*np.log(M)*rot_exp_T + rot_select_1)*K + 2*rot_COEF)*3 + rot_exp_T )*r
        epsilon_SS = epsilon_S / num_rotations
        
        num_Subprepare = 2*3*K*3*r
        epsilon_mus = epsilon_mu / num_Subprepare
        
        Subprep = 4*J + 4*np.log(1/epsilon_mus) + 8*np.log2(1/epsilon_SS) + 12*np.log2(J)
        n = 1/3*np.log2(N) + 1
        Prep  = 3*(79*n**2 +43*n*np.log2(M0) + 44*n)
        exp_T = rot_exp_T * 4*np.log(1/epsilon_SS)
        select_0 = 16*eta*np.log2(N)
        select_1 = 8*eta*np.log2(N) + 14*(np.log2(N))**2 + 4*np.log2(N)*np.log(1/epsilon_SS)
        
        HAM_T = 2*np.log(M)*exp_T + 2*(3*(Subprep + Prep)) + select_0 + select_1 #The 3 multiplying Subprep and Prep comes from oblivious AA
        U = 8*(np.log2(M) + np.log2(1/epsilon_SS))
        ADD = 4*np.log2(K)
        Comp = 8*np.log2(M)
        
        COEF = rot_COEF * (10 + 12*np.log2(K))
        REF = 16*(np.log2(Gamma) + 2*np.log(K+1)+ 2*np.log(M))
        
        cost = (((4*K*U + K*Comp + (3*K + 1)*ADD + K*HAM_T) + 2*COEF)*3  + 2*REF)*r
        
        antisymmetrization = 3*eta*np.log2(eta)*(np.log2(eta)-1)*(2* np.ceil(np.log2(eta**2)) + np.log(N))
        
        return cost + antisymmetrization