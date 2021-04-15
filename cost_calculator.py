import trotter_based_methods
import taylor_based_methods
import plane_waves_methods
import qrom_methods
import interaction_picture
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

class Cost_calculator:

    def __init__(self, molecule, tools):

        self.molecule = molecule
        self.tools = tools
        self.costs = {}

    def calculate_cost(self, method, ao_labels = None): 


        if method == 'qdrift' or method == 'rand_ham':

            methods_trotter = trotter_based_methods.Trotter_based_methods()

            if method == 'qdrift': self.costs['qdrift'] = methods_trotter.calc_qdrift_resources(
                                        self.molecule.lambda_value, 
                                        self.molecule.N, 
                                        deltaE = 1e-4, 
                                        P_failure = .1)

            elif method == 'rand_ham': self.costs['rand_ham'] = methods_trotter.calc_rand_ham_resources(
                                            self.molecule.Lambda_value, 
                                            self.molecule.lambda_value, 
                                            self.molecule.gamma, 
                                            self.molecule.N, 
                                            deltaE = 1e-4, 
                                            P_failure = .1)
        
        elif method == 'taylor_naive' or method == 'taylor_on_the_fly' or method == 'configuration_interaction':

            methods_taylor = taylor_based_methods.Taylor_based_methods(self.tools)

            errors = 3 # epsilon_PEA, epsilon_HS, epsilon_S
            constraint = NonlinearConstraint(fun=self.tools.sum_constraint, lb=0, ub=0.015)
            bounds = [(0, 0.015) for _ in range(errors)]

            res = minimize(
                methods_taylor.taylor_naive,
                x0=np.zeros(errors),
                args=(self.molecule.Lambda_value, self.molecule.Gamma, self.N,),
                constraints=constraint,
                bounds=bounds
            )

            errors = {"epsilon_PEA": res.x[0], "epsilon_HS": res.x[1], "epsilon_S": res.x[2]}

            if method == 'taylor_naive':
                self.costs['taylor_naive'] = methods_taylor.taylor_naive(
                                                self.molecule.Lambda_value,
                                                self.molecule.Gamma,
                                                self.molecule.N,
                                                errors['epsilon_PEA'],
                                                errors['epsilon_HS'],
                                                errors['epsilon_S'])

            '''
            elif method == 'taylor_on_the_fly':
                zeta_max_i = self.molecule.calculate_zeta_i_max()
                self.costs['taylor_on_the_fly'] = methods_taylor.taylor_on_the_fly(Gamma, N, phi_max, dphi_max, epsilon_PEA, epsilon_HS, epsilon_S, epsilon_H, zeta_max_i = zeta_max_i, epsilon_tay)
            elif method == 'configuration_interaction':
                phi_max, dphi_max = self.molecule.molecular_orbital_parameters()
                # alpha, gamma1, gamma2 are used to calculate K0, K1, K2 (see eq D14 in overleaf)
                self.costs['configuration_interaction'] = methods_taylor.configuration_interaction(N, eta, alpha, gamma1, K0, K1, K2, epsilon_PEA, epsilon_HS, epsilon_S, epsilon_H, epsilon_tay, zeta_max_i, phi_max = phi_max, dphi_max = dphi_max)

        elif method == 'low_depth_trotter' or method == 'low_depth_taylor' or method == 'low_depth_taylor_on_the_fly':

            methods_plane_waves = plane_waves_methods.Plane_waves_methods(self.tools)

            # This methods are plane waves, so instead of calling self.molecule.get_basic_parameters() one should call self.molecule.build_grid()
            # grid_length is the only parameter of build_grid. Should be calculated such that the number of basis functions
            #   is ~= 100*self.molecule_data.n_orbitals*2. grid_length ~= int(np.cbrt(100*self.molecule.molecule_data.n_orbitals * 2))
            # Omega is returned by self.molecule.build_grid()
            # J = len(self.molecule.geometry) #is the number of atoms in the molecule

            if method == 'low_depth_trotter':
                self.costs['low_depth_trotter'] = methods_plane_waves.low_depth_trotter(N, eta, Omega, epsilon_PEA, epsilon_HS, epsilon_S)
            elif method == 'low_depth_taylor':
                self.costs['low_depth_taylor'] = methods_plane_waves.low_depth_taylor(N, lambda_value, Lambda_value, epsilon_PEA, epsilon_HS, epsilon_S, Ham_norm)
            elif method == 'low_depth_taylor_on_the_fly':
                # find x_max from cell volume assuming a perfect cube centered on 0
                self.costs['low_depth_taylor_on_the_fly'] = methods_plane_waves.low_depth_taylor_on_the_fly(N, eta, lambda_value, Omega, epsilon_PEA, epsilon_HS, epsilon_S, epsilon_tay, Ham_norm, J, x_max)

        elif method == 'linear_t' or method == 'sparsity_low_rank':

            methods_qrom = qrom_methods.QROM_methods()

            if method == 'linear_t':
                self.costs['linear_t'] = methods_qrom.linear_T(N, lambda_value, epsilon_PEA, epsilon_S, Ham_norm)
            elif method == 'sparsity_low_rank':
                #todo: how to select the sparsify option here appropriately
                molecular_hamiltonian, final_rank = self.molecule.low_rank_approximation(occupied_indices = self.molecule.occupied_indices, active_indices = self.molecule.active_indices, virtual_indices = self.molecule.virtual_indices, sparsify = True)
                molecule.get_basic_parameters(molecular_hamiltonian = molecular_hamiltonian)
                self.costs['sparsity_low_rank'] = methods_qrom.sparsity_low_rank(N = self.molecule.N, lambda_value = self.molecule.lambd, eps_PEA, eps_S, L = final_rank, Ham_norm)
        
        elif method == 'interaction_picture' or method == 'sublinear_scaling':

            methods_interaction_picture = interaction_picture.Interaction_picture()

            if method == 'interaction_picture':
                self.costs['interaction_picture'] = methods_interaction_picture.interaction_picture(N, Gamma, lambda_value_T, lambda_value_U_V, epsilon_S, epsilon_HS, epsilon_PEA)
            elif method == 'sublinear_scaling':
                self.costs['sublinear_scaling'] = methods_interaction_picture.sublinear_scaling_interaction(N, eta, Gamma, lambda_value_T, lambda_value_U_V, epsilon_S, epsilon_HS, epsilon_PEA, epsilon_mu, epsilon_M_0, J)
                '''