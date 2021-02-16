import numpy as np

class Interaction_picture:

    def __init__(self):
        pass

    def interaction_picture(self, N, Gamma, lambd_T, lambd_U_V, eps_S, eps_HS, eps_PEA):
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
        K = np.ceil( -1  + 2* np.log(2*r/eps_HS)/np.log(np.log(2*r/eps_HS))) # We 
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
        exp_U_V= 46*N*(np.log(1/epsilon_SS))**2+8*N + 8*N*np.log2(1/epsilon_SS)*np.log2(N) + 4*N*np.log(N)
        COEF = rot_COEF * (10 + 12*np.log2(K))
        U = 8*(np.log2(M) + np.log2(1/epsilon_SS))
        ADD = 4*np.log2(K)
        Comp = 8*np.log2(M)
        FFFT = (2 + 4*np.log(1/epsilon_SS))*N*np.log2(N) - 4*np.log2(1/epsilon_SS)*N
        Prep = 2**9*(1 + np.log2(N))+2**6*3*N*np.log2(1/epsilon_SS)
        Select = 8*N
        REF = 16*(np.log2(Gamma) + 2*np.log(K+1)+ 2*np.log(M))
        
        cost = ((((2*Prep + Select + 2*FFFT + 2*np.log(M)*exp_U_V)*K + (3*K+1)*ADD + K*Comp + 4*K*U +2*COEF)*3 + 2*REF) + exp_U_V)*r
        
        return cost

    ## Sublinear scaling and interaction picture babbush2019quantum
    def sublinear_scaling_interaction(self, N, eta, Gamma, lambd_T, lambd_U_V, eps_S, eps_HS, eps_PEA, eps_mu, eps_M_0, J):
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
        K = np.ceil( -1  + 2* np.log(2*r/eps_HS)/np.log(np.log(2*r/eps_HS))) # We 
        delta = eps_HS / t # Alternatively we can substitute t by r changing delta in the following line to 1/2. t represents L in the main text (see before eq 21 in the original article)
        tau = 1/np.ceil(2*lambd_U_V) # tau = t/ np.ceil(2 * lambd_T * t)
        M = np.max(16*tau/delta * (lambd_U_V + 2*lambd_T), K**2)
        M0 = eta * N * tau / (eps_M_0/r)
        
        rot_exp_T = np.log2(eta) + 2*np.log2(N)
        rot_select_1 = 1/3*np.log2(N) + 2
        rot_Subprepare = 2 # Only the two rotations from Uniform in Subprepare
        rot_COEF = 2**(np.ceil(np.log2(K) + 1))
        
        num_rotations = (((2*np.log(M)*rot_exp_T + rot_select_1)*K + 2*rot_COEF)*3 + rot_exp_T )*r
        eps_SS = eps_S / num_rotations
        
        num_Subprepare = 2*3*K*3*r
        eps_mus = eps_mu / num_Subprepare
        
        Subprep = 4*J + 4*np.log(1/eps_mus) + 8*np.log2(1/eps_SS) + 12*np.log2(J)
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