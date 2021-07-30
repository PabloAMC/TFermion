from scipy import optimize
import trotter_based_methods
import taylor_based_methods
import plane_waves_methods
import qrom_methods
import interaction_picture
import numpy as np
from scipy.optimize import minimize

class Cost_calculator:

    def __init__(self, molecule, tools):

        self.molecule = molecule
        self.tools = tools
        self.costs = {}

    def calculate_cost(self, method, ao_labels = None): 


        if method == 'qdrift' or method == 'rand_ham':
            #todo: error handling will need to be modified after I talk with Michael

            methods_trotter = trotter_based_methods.Trotter_based_methods(self.tools)

            # calculate the basis of the molecule (and its parameters)
            self.molecule.get_basic_parameters()

            if method == 'qdrift': 
                
                lambda_value = self.molecule.lambda_value
                arguments = (lambda_value)

                # generate values for errors epsilon_PEA, epsilon_HS, epsilon_S
                optimized_errors = self.calculate_optimized_errors(3, methods_trotter.calc_qdrift_resources, arguments)
                
                
                self.costs['qdrift'] = methods_trotter.calc_qdrift_resources(
                                        optimized_errors.x,
                                        lambda_value)

            elif method == 'rand_ham': 

                Lambda_value = self.molecule.Lambda_value
                Gamma = self.molecule.Gamma

                arguments = (Lambda_value, Gamma)

                # generate values for errors epsilon_PEA, epsilon_HS, epsilon_S
                optimized_errors = self.calculate_optimized_errors(3, methods_trotter.calc_rand_ham_resources, arguments)
                
                self.costs['rand_ham'] = methods_trotter.calc_rand_ham_resources(
                                            optimized_errors.x,
                                            Lambda_value,
                                            Gamma)
        
        elif method == 'taylor_naive' or method == 'taylor_on_the_fly' or method == 'configuration_interaction':

            methods_taylor = taylor_based_methods.Taylor_based_methods(self.tools)

            # calculate the basis of the molecule (and its parameters)
            self.molecule.get_basic_parameters()

            if method == 'taylor_naive':

                arguments = (self.molecule.Lambda_value, self.molecule.Gamma, self.molecule.N)

                # generate values for errors epsilon_PEA, epsilon_HS, epsilon_S
                optimized_errors = self.calculate_optimized_errors(3, methods_taylor.taylor_naive, arguments)

                self.costs['taylor_naive'] = methods_taylor.taylor_naive(
                    optimized_errors.x,
                    self.molecule.Lambda_value,
                    self.molecule.Gamma,
                    self.molecule.N)


            elif method == 'taylor_on_the_fly':

                phi_max, dphi_max, _, _ = self.molecule.molecular_orbital_parameters()
                zeta_max_i = self.molecule.calculate_zeta_i_max()
                J = len(self.molecule.molecule_geometry) #is the number of atoms in the molecule

                arguments = (self.molecule.N, self.molecule.lambda_value, self.molecule.Lambda_value, self.molecule.Gamma, phi_max, dphi_max, zeta_max_i, J)

                # generate values for errors epsilon_PEA, epsilon_HS, epsilon_S, eps_H, eps_taylor
                optimized_errors = self.calculate_optimized_errors(5, methods_taylor.taylor_on_the_fly, arguments)

                self.costs['taylor_on_the_fly'] = methods_taylor.taylor_on_the_fly(
                    optimized_errors.x,
                    self.molecule.N,
                    self.molecule.lambda_value,
                    self.molecule.Lambda_value,
                    self.molecule.Gamma,
                    phi_max,
                    dphi_max,
                    zeta_max_i,
                    J)
            
            elif method == 'configuration_interaction':
                phi_max, _, grad_max, _ = self.molecule.molecular_orbital_parameters()
                N = self.molecule.N
                x_max = self.molecule.xmax
                alpha = self.molecule.min_alpha()
                gamma1 = grad_max * x_max / phi_max
                gamma2 = grad_max**2 * x_max**2 / phi_max
                eta = self.molecule.eta
                zeta_max_i = self.molecule.calculate_zeta_i_max()
                J = len(self.molecule.molecule_geometry) #is the number of atoms in the molecule

                arguments = (N, eta, alpha, gamma1, gamma2, zeta_max_i, phi_max, J)

                # generate values for errors epsilon_PEA, epsilon_HS, epsilon_S, eps_H, eps_taylor
                optimized_errors = self.calculate_optimized_errors(5, methods_taylor.configuration_interaction, arguments)

                # alpha, gamma1, gamma2 are used to calculate K0, K1, K2 (see eq D14 in overleaf)
                self.costs['configuration_interaction'] = methods_taylor.configuration_interaction(
                    optimized_errors.x,
                    N,
                    eta,    
                    alpha,
                    gamma1,
                    gamma2,
                    zeta_max_i,
                    phi_max,
                    J)

        
        elif method == 'low_depth_trotter' or method == 'low_depth_taylor' or method == 'low_depth_taylor_on_the_fly':

            methods_plane_waves = plane_waves_methods.Plane_waves_methods(self.tools)

            # This methods are plane waves, so instead of calling self.molecule.get_basic_parameters() one should call self.molecule.build_grid()
            # grid_length is the only parameter of build_grid. Should be calculated such that the number of basis functions
            #   is ~= 100*self.molecule_data.n_orbitals*2. grid_length ~= int(np.cbrt(100*self.molecule.molecule_data.n_orbitals * 2))
            # Omega is returned by self.molecule.build_grid()
            # J = len(self.molecule.geometry) #is the number of atoms in the molecule

            if method == 'low_depth_trotter':

                grid_length = int((self.molecule.N * 100) ** 1/3)
                grid_length = 5
                Omega = self.molecule.build_grid(grid_length).volume

                N = self.molecule.N
                eta = self.molecule.eta

                arguments = (N, eta, Omega)

                # generate values for errors epsilon_PEA, epsilon_HS, epsilon_S
                optimized_errors = self.calculate_optimized_errors(3, methods_plane_waves.low_depth_trotter, arguments)

                self.costs['low_depth_trotter'] = methods_plane_waves.low_depth_trotter(
                    optimized_errors.x,
                    N, 
                    eta, 
                    Omega)

            elif method == 'low_depth_taylor':

                grid_length = int((self.molecule.N * 100) ** 1/3)
                grid_length = 5
                Omega = self.molecule.build_grid(grid_length).volume

                N = self.molecule.N
                lambda_value = self.molecule.lambda_value
                Lambda_value = self.molecule.Lambda_value
                H_norm_lambda_ratio = self.molecule.H_norm_lambda_ratio 

                arguments = (N, lambda_value, Lambda_value, H_norm_lambda_ratio)

                # generate value for errors epsilon_PEA, epsilon_HS, epsilon_S
                optimized_errors = self.calculate_optimized_errors(3, methods_plane_waves.low_depth_taylor, arguments)

                self.costs['low_depth_taylor'] = methods_plane_waves.low_depth_taylor(
                    optimized_errors.x,
                    N, 
                    lambda_value, 
                    Lambda_value, 
                    H_norm_lambda_ratio)

            elif method == 'low_depth_taylor_on_the_fly':

                grid_length = int((self.molecule.N * 100) ** 1/3)
                grid_length = 5
                Omega = self.molecule.build_grid(grid_length).volume

                N = self.molecule.N
                eta = self.molecule.eta
                Gamma = self.molecule.Gamma
                lambda_value = self.molecule.lambda_value
                
                x_max = self.molecule.xmax
                J = len(self.molecule.molecule_geometry) #is the number of atoms in the molecule

                arguments = (N, eta, Gamma, lambda_value, Omega, J, x_max)

                # generate value for errors epsilon_PEA, epsilon_HS, epsilon_S, epsilon_H, epsilon_tay
                optimized_errors = self.calculate_optimized_errors(5, methods_plane_waves.low_depth_taylor_on_the_fly, arguments)

                # find x_max from cell volume assuming a perfect cube centered on 0
                self.costs['low_depth_taylor_on_the_fly'] = methods_plane_waves.low_depth_taylor_on_the_fly(
                    optimized_errors.x,
                    N, 
                    eta,
                    Gamma,
                    lambda_value, 
                    Omega,
                    J, 
                    x_max)

        elif method == 'linear_t' or method == 'sparsity_low_rank':

            methods_qrom = qrom_methods.QROM_methods(self.tools)

            if method == 'linear_t':

                grid_length = int((self.molecule.N * 100) ** 1/3)
                grid_length = 5
                Omega = self.molecule.build_grid(grid_length).volume

                
                N = self.molecule.N
                lambda_value = self.molecule.lambda_value
                H_norm_lambda_ratio = self.molecule.H_norm_lambda_ratio 

                arguments = (N, lambda_value, H_norm_lambda_ratio)

                # generate value for errors epsilon_PEA, epsilon_S
                optimized_errors = self.calculate_optimized_errors(2, methods_qrom.linear_T, arguments)
                
                self.costs['linear_t'] = methods_qrom.linear_T(
                    optimized_errors.x,
                    N, 
                    lambda_value,
                    H_norm_lambda_ratio)

            elif method == 'sparsity_low_rank':
                
                #todo: how to select the sparsify option here appropriately
                molecular_hamiltonian, final_rank = self.molecule.low_rank_approximation(sparsify = True)
                self.molecule.get_basic_parameters(molecular_hamiltonian = molecular_hamiltonian)

                N = self.molecule.N
                lambda_value = self.molecule.lambda_value
                H_norm_lambda_ratio = self.molecule.H_norm_lambda_ratio
                sparsity_d = self.molecule.sparsity_d 

                arguments = (N, lambda_value, final_rank, H_norm_lambda_ratio, sparsity_d)
                
                # generate value for errors epsilon_PEA, epsilon_S
                optimized_errors = self.calculate_optimized_errors(2, methods_qrom.sparsity_low_rank, arguments)

                self.costs['sparsity_low_rank'] = methods_qrom.sparsity_low_rank(
                    optimized_errors.x,
                    N, 
                    lambda_value,
                    final_rank, 
                    H_norm_lambda_ratio,
                    sparsity_d)
        
        elif method == 'interaction_picture' or method == 'sublinear_scaling':

            methods_interaction_picture = interaction_picture.Interaction_picture(self.tools)

            if method == 'interaction_picture':

            
                grid_length = int((self.molecule.N * 100) ** 1/3)
                grid_length = 5
                lambda_value_T, lambda_value_U_V = self.molecule.lambda_of_Hamiltonian_terms_2nd(self.molecule.build_grid(grid_length))

                N = self.molecule.N
                Gamma = self.molecule.Gamma

                arguments = (N, Gamma, lambda_value_T, lambda_value_U_V)

                # generate value for errors epsilon_S, epsilon_HS, epsilon_PEA
                optimized_errors = self.calculate_optimized_errors(3, methods_interaction_picture.interaction_picture, arguments)

                self.costs['interaction_picture'] = methods_interaction_picture.interaction_picture(
                    optimized_errors.x,
                    N, 
                    Gamma, 
                    lambda_value_T, 
                    lambda_value_U_V)
            
            
            
            elif method == 'sublinear_scaling':

                grid_length = int((self.molecule.N * 100) ** 1/3)
                grid_length = 5
                self.molecule.build_grid(grid_length).volume

                N = self.molecule.N
                eta = self.molecule.eta
                Gamma = self.molecule.Gamma
                
                grid_length = int((self.molecule.N * 100) ** 1/3)
                grid_length = 5
                Omega = self.molecule.build_grid(grid_length).volume

                J = len(self.molecule.molecule_geometry) #is the number of atoms in the molecule

                Omega = self.molecule.Omega
                lambda_value_T, lambda_value_U_V = self.molecule.lambda_of_Hamiltonian_terms_1st(eta, Omega, N)

                arguments = (N, eta, Gamma, lambda_value_T, lambda_value_U_V, J)

                # generate value for errors epsilon_S, epsilon_HS, epsilon_PEA, epsilon_mu, epsilon_M_0, epsilon_R
                optimized_errors = self.calculate_optimized_errors(6, methods_interaction_picture.sublinear_scaling_interaction, arguments)

                self.costs['sublinear_scaling'] = methods_interaction_picture.sublinear_scaling_interaction(
                    optimized_errors.x,
                    N, 
                    eta, 
                    Gamma, 
                    lambda_value_T, 
                    lambda_value_U_V,
                    J)

        else:
            print('<*> ERROR: method', method, 'not implemented or not existing')

    def calculate_optimized_errors(self, number_errors, cost_method, arguments):

        constraints = self.tools.generate_constraints(number_errors)
        initial_values = self.tools.generate_initial_error_values(number_errors)

        optimized_errors = minimize(
            fun=cost_method,
            x0=initial_values,
            method=self.tools.config_variables['optimization_method'],
            constraints=constraints,
            args=arguments,
        )

        return optimized_errors