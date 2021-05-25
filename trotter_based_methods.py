import numpy as np

class Trotter_based_methods:

    def __init__(self, tools):

        self.tools = tools

    # qDrift and Trotterization

    def calc_qdrift_resources(self, epsilons, lambd):

        epsilon_PEA = epsilons[0]
        epsilon_HS = epsilons[1]
        epsilon_S = epsilons[2]

        deltaE = epsilon_PEA
        eps_tot = epsilon_HS

        P_failure = 6*eps_tot

        n = ((27*np.pi**2/2)*(lambd/deltaE)**2) / P_failure**3

        epsilon_SS = epsilon_S/(2*n) # The 2 is due to the control

        rost_cost_factor = 2*self.tools.c_z_rotation(epsilon_SS)

        return 1/(1-P_failure)*rost_cost_factor*n

    def calc_rand_ham_resources(self, epsilons, Lambd, Gamma):

        epsilon_PEA = epsilons[0]
        epsilon_HS = epsilons[1]
        epsilon_S = epsilons[2]

        deltaE = epsilon_PEA
        eps_tot = epsilon_HS

        P_failure = 8*eps_tot

        n = 4.35*np.sqrt(8)*(np.pi*Lambd/deltaE)**(3/2) *(Gamma/ P_failure)**2

        epsilon_SS = epsilon_S/(2*n) # The 2 is due to the control

        rost_cost_factor = 2*self.tools.c_z_rotation(epsilon_SS)

        return 1/(1-P_failure)*rost_cost_factor*n
