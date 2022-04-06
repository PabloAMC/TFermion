import numpy as np

class Trotter_based_methods:

    def __init__(self, tools):

        self.tools = tools

    # qDrift and Trotterization

    def calc_qdrift_resources(self, epsilons, p_fail, lambda_value):

        epsilon_PEA = epsilons[0]
        epsilon_HS = epsilons[1]
        epsilon_S = epsilons[2]

        deltaE = epsilon_PEA
        eps_tot = epsilon_HS

        n = (np.pi*lambda_value/deltaE)**2 *(1/eps_tot) * ((1+p_fail)/p_fail)**2 #eq 42 in the original paper

        epsilon_SS = epsilon_S/(2*n) # The 2 is due to the control

        rot_cost_factor = self.tools.c_pauli_rotation_synthesis(epsilon_SS)

        return rot_cost_factor*n

    def calc_rand_ham_resources(self, epsilons, p_fail, Lambda_value, Gamma):

        epsilon_PEA = epsilons[0]
        epsilon_HS = epsilons[1]
        epsilon_S = epsilons[2]

        deltaE = epsilon_PEA
        eps_tot = epsilon_HS

        P_failure = 8*eps_tot # From P_f = p_fail + 2*eps_tot and p_fail = (3/4)*P_f

        n = 8*Gamma**2 *(np.pi*Lambda_value/(2*deltaE))**(3/2) * (1/eps_tot) * ((1+P_failure)/P_failure)**(3/2) #eq 54 in the original paper

        epsilon_SS = epsilon_S/(2*n) # The 2 is due to the control

        rot_cost_factor = self.tools.c_pauli_rotation_synthesis(epsilon_SS)

        return rot_cost_factor*n
