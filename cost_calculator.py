import trotter_based_methods
import taylor_based_methods
import plane_waves_methods
import qrom_methods
import interaction_picture

class Cost_calculator:

    def __init__(self, molecule, tools):

        self.molecule = molecule
        self.tools = tools
        self.costs = {}

    def calculate_cost(self, method): 


        if method == 'qdrift' or method == 'rand_ham':

            methods_trotter = trotter_based_methods.Trotter_based_methods()
            
            if method == 'qdrift':
                self.costs['q_drift'] = methods_trotter.calc_qdrift_resources(lambda_value, N, deltaE = 1e-4, P_failure = .1)
            elif method == 'rand_ham':
                self.costs['rand_ham'] = methods_trotter.calc_rand_ham_resources(Lambda_value, lambda_value, Gamma, N, deltaE = 1e-4, P_failure = .1)

        elif method == 'taylor_naive' or method == 'taylor_on_the_fly' or method == 'configuration_interaction':

            methods_taylor = taylor_based_methods.Taylor_based_methods(self.tools)

            if method == 'taylor_naive':
                self.costs['taylor_naive'] = methods_taylor.taylor_naive(Lambda_value, Gamma, N, epsilon_PEA, epsilon_HS, epsilon_S)
            elif method == 'taylor_on_the_fly':
                self.costs['taylor_on_the_fly'] = methods_taylor.taylor_on_the_fly(Gamma, N, phi_max, dphi_max, zeta_max_i, epsilon_PEA, epsilon_HS, epsilon_S, epsilon_H, max_zeta_i, eps_tay)
            elif method == 'configuration_interaction':
                self.costs['configuration_interaction'] = methods_taylor.configuration_interaction(N, eta, alpha, gamma1, K0, K1, K2, epsilon_PEA, epsilon_HS, epsilon_S, epsilon_H, eps_tay, max_zeta_i, phi_max, dphi_max)

        elif method == 'low_depth_trotter' or method == 'low_depth_taylor' or method == 'low_depth_taylor_on_the_fly':

            methods_plane_waves = plane_waves_methods.Plane_waves_methods(self.tools)

            if method == 'low_depth_trotter':
                self.costs['low_depth_trotter'] = methods_plane_waves.low_depth_trotter(N, eta, Omega, eps_PEA, eps_HS, eps_S)
            elif method == 'low_depth_taylor':
                self.costs['low_depth_taylor'] = methods_plane_waves.low_depth_taylor(N, lambda_value, Lambda_value, eps_PEA, eps_HS, eps_S, Ham_norm)
            elif method == 'low_depth_taylor_on_the_fly':
                self.costs['low_depth_taylor_on_the_fly'] = methods_plane_waves.low_depth_taylor_on_the_fly(N, eta, lambda_value, Omega, eps_PEA, eps_HS, eps_S, eps_tay, Ham_norm, J, x_max)

        elif method == 'linear_t' or method == 'sparsity_low_rank':

            methods_qrom = qrom_methods.QROM_methods()

            if method == 'linear_t':
                self.costs['linear_t'] = methods_qrom.linear_T(N, lambda_value, eps_PEA, eps_S, Ham_norm)
            elif method == 'sparsity_low_rank':
                self.costs['sparsity_low_rank'] = methods_qrom.sparsity_low_rank(N, lambda_value, eps_PEA, eps_S, L, Ham_norm)
        
        elif method == 'interaction_picture' or method == 'sublinear_scaling':

            methods_interaction_picture = interaction_picture.Interaction_picture()

            if method == 'interaction_picture':
                self.costs['interaction_picture'] = methods_interaction_picture.interaction_picture(N, Gamma, lambda_value_T, lambda_value_U_V, eps_S, eps_HS, eps_PEA)
            elif method == 'sublinear_scaling':
                self.costs['sublinear_scaling'] = methods_interaction_picture.sublinear_scaling_interaction(N, eta, Gamma, lambda_value_T, lambda_value_U_V, eps_S, eps_HS, eps_PEA, eps_mu, eps_M_0, J)