import numpy as np

class Trotter_based_methods:

    def __init__(self):
        pass

    # qDrift and Trotterization

    def calc_qdrift_resources(self, lambd, weighted_average_paulis, epsilon_PEA, epsilon_HS, epsilon_S):

        deltaE = epsilon_PEA
        eps_tot = epsilon_HS

        P_failure = 6*eps_tot

        n = ((27*np.pi**2/2)*(lambd/deltaE)**2) / P_failure**3

        epsilon_SS = epsilon_S/(n*weighted_average_paulis) # from eq 39 same as the one above

        rost_cost_factor = weighted_average_paulis*self.tools.c_rotation_synthesis(epsilon_SS)
        
        return 1/(1-P_failure)*rost_cost_factor*n

    def calc_rand_ham_resources(self, Lambd, Gamma, average_paulis, epsilon_PEA, epsilon_HS, epsilon_S):

        deltaE = epsilon_PEA
        eps_tot = epsilon_HS

        P_failure = 8*eps_tot

        n = 4.35*np.sqrt(8)*(np.pi*Lambd/deltaE)**(3/2) *(Gamma/ P_failure)**2

        epsilon_SS = epsilon_S/(n*average_paulis)

        rost_cost_factor = average_paulis*self.tools.c_rotation_synthesis(epsilon_SS)
        
        return 1/(1-P_failure)*rost_cost_factor*n
