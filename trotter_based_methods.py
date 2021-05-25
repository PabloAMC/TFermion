import numpy as np

class Trotter_based_methods:

    def __init__(self):
        pass

    # qDrift and Trotterization

    def calc_qdrift_resources(self, lambd, weighted_avg_Z_per_unitary, weighted_avg_XY_per_unitary, epsilon_PEA, epsilon_HS, epsilon_S):

        deltaE = epsilon_PEA
        eps_tot = epsilon_HS

        P_failure = 6*eps_tot

        n = ((27*np.pi**2/2)*(lambd/deltaE)**2) / P_failure**3

        epsilon_SS = epsilon_S/(2*n) # The 2 is due to the control

        rost_cost_factor = 2*self.tools.c_z_rotation_synthesis(epsilon_SS)

        return 1/(1-P_failure)*rost_cost_factor*n

    def calc_rand_ham_resources(self, Lambd, Gamma, avg_Z_per_unitary, avg_XY_per_unitary, epsilon_PEA, epsilon_HS, epsilon_S):

        deltaE = epsilon_PEA
        eps_tot = epsilon_HS

        P_failure = 8*eps_tot

        n = 4.35*np.sqrt(8)*(np.pi*Lambd/deltaE)**(3/2) *(Gamma/ P_failure)**2

        epsilon_SS = epsilon_S/(2*n) # The 2 is due to the control

        rost_cost_factor = 2*self.tools.c_z_rotation_synthesis(epsilon_SS)

        return 1/(1-P_failure)*rost_cost_factor*n
