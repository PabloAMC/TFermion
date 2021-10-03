from scipy import optimize
import trotter_based_methods
import taylor_based_methods
import plane_waves_methods
import qrom_methods
import interaction_picture
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

class Cost_calculator:

    def __init__(self, molecule, tools, molecule_info_type):

        self.molecule = molecule
        self.tools = tools
        self.molecule_info_type = molecule_info_type
        self.costs = {'qdrift': [],
                    'rand_ham': [],
                    'taylor_naive': [],
                    'taylor_on_the_fly': [],
                    'configuration_interaction': [],
                    'low_depth_trotter': [],
                    'low_depth_taylor': [],
                    'low_depth_taylor_on_the_fly': [],
                    'linear_t': [],
                    'sparsity_low_rank': [],
                    'interaction_picture': []
                    }
        self.basis = self.tools.config_variables['basis']
        self.runs = self.tools.config_variables['runs']

    def calculate_cost(self, method): 

        if self.molecule_info_type == 'name':
            
            json_name = str(self.molecule.molecule_info)+ '_' +  str(self.basis)
            self.molecule.load(json_name = 'parameters/'+json_name)

        '''
        if method in ['low_depth_trotter','low_depth_taylor','low_depth_taylor_on_the_fly','linear_t','interaction_picture']:
            b = 'plane'
            # we add the gaussian basis to the name because it is important to figure out the comparable number of plane wave basis functions
            self.molecule.load(json_name = str(self.molecule.molecule_info)+ '_' +  str(self.basis)+ '_' +b)

        elif method in ['rand_ham','qdrift','naive_taylor','taylor_on_the_fly','configuration_interaction','sparsity_low_rank']:
            b = 'gauss'
            self.molecule.load(json_name = str(self.molecule.molecule_info)+ '_' + str(self.basis)+ '_' +b)

        '''
        if method == 'qdrift' or method == 'rand_ham':
            #todo: error handling will need to be modified after I talk with Michael

            methods_trotter = trotter_based_methods.Trotter_based_methods(self.tools)

            # calculate the basis of the molecule (and its parameters)
            if not hasattr(self.molecule, 'lambda_value') or not hasattr(self.molecule, 'Lambda_value') or not hasattr(self.molecule, 'eta') or not hasattr(self.molecule, 'Gamma'):
                self.molecule.get_basic_parameters()

            if method == 'qdrift': 
                
                lambda_value = self.molecule.lambda_value
                arguments = (lambda_value)

                # generate values for errors epsilon_PEA, epsilon_HS, epsilon_S
                for _ in range(self.runs):
                    optimized_errors = self.calculate_optimized_errors(3, methods_trotter.calc_qdrift_resources, arguments)
                    
                    
                    self.costs['qdrift'] += [methods_trotter.calc_qdrift_resources(
                                            optimized_errors.x,
                                            lambda_value)]

            elif method == 'rand_ham': 

                Lambda_value = self.molecule.Lambda_value
                Gamma = self.molecule.Gamma

                arguments = (Lambda_value, Gamma)

                # generate values for errors epsilon_PEA, epsilon_HS, epsilon_S
                for _ in range(self.runs):
                    optimized_errors = self.calculate_optimized_errors(3, methods_trotter.calc_rand_ham_resources, arguments)
                    
                    self.costs['rand_ham'] += [methods_trotter.calc_rand_ham_resources(
                                                optimized_errors.x,
                                                Lambda_value,
                                                Gamma)]
        
        elif method == 'taylor_naive' or method == 'taylor_on_the_fly' or method == 'configuration_interaction':

            methods_taylor = taylor_based_methods.Taylor_based_methods(self.tools)

            # calculate the basis of the molecule (and its parameters)
            if not hasattr(self.molecule, 'lambda_value') or not hasattr(self.molecule, 'Lambda_value') or not hasattr(self.molecule, 'eta') or not hasattr(self.molecule, 'Gamma'):
                self.molecule.get_basic_parameters()

            lambda_value = self.molecule.lambda_value
            Lambda_value = self.molecule.Lambda_value
            Gamma = self.molecule.Gamma
            N = self.molecule.N

            if method == 'taylor_naive':

                arguments = (lambda_value, Gamma, N)

                # generate values for errors epsilon_PEA, epsilon_HS, epsilon_S
                for _ in range(self.runs):
                    optimized_errors = self.calculate_optimized_errors(3, methods_taylor.taylor_naive, arguments)

                    self.costs['taylor_naive'] += [methods_taylor.taylor_naive(
                        optimized_errors.x,
                        lambda_value,
                        Gamma,
                        N)]


            elif method == 'taylor_on_the_fly':

                if not hasattr(self.molecule, 'phi_max') or not hasattr(self.molecule, 'dphi_max'):
                    self.molecule.molecular_orbital_parameters()
                if not hasattr(self.molecule, 'zeta_max_i'):
                    self.molecule.calculate_zeta_max_i()

                zeta_max_i = self.molecule.zeta_max_i
                phi_max = self.molecule.phi_max
                dphi_max = self.molecule.dphi_max
                J = len(self.molecule.molecule_geometry) #is the number of atoms in the molecule

                arguments = (N, lambda_value, Lambda_value, Gamma, phi_max, dphi_max, zeta_max_i, J)

                # generate values for errors epsilon_PEA, epsilon_HS, epsilon_S, eps_H, eps_taylor
                for _ in range(self.runs):
                    optimized_errors = self.calculate_optimized_errors(5, methods_taylor.taylor_on_the_fly, arguments)

                    self.costs['taylor_on_the_fly'] += [methods_taylor.taylor_on_the_fly(
                        optimized_errors.x,
                        N,
                        lambda_value,
                        Lambda_value,
                        Gamma,
                        phi_max,
                        dphi_max,
                        zeta_max_i,
                        J)]
            
            elif method == 'configuration_interaction':
                if not hasattr(self.molecule, 'phi_max') or not hasattr(self.molecule, 'grad_max') or not hasattr(self.molecule, 'lapl_max'):
                    self.molecule.molecular_orbital_parameters()
                if not hasattr(self.molecule, 'alpha'):
                    self.molecule.min_alpha()
                if not hasattr(self.molecule, 'zeta_max_i'):
                    self.molecule.calculate_zeta_max_i()

                N = self.molecule.N # computed from initialising the molecule
                x_max = 1 # Default units are Angstroms. See https://en.wikipedia.org/wiki/Atomic_radius and https://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page)
                phi_max = self.molecule.phi_max
                alpha = self.molecule.alpha
                eta = self.molecule.eta
                zeta_max_i = self.molecule.zeta_max_i

                gamma1 = self.molecule.grad_max * x_max / self.molecule.phi_max
                gamma2 = self.molecule.lapl_max * x_max**2 / self.molecule.phi_max

                J = len(self.molecule.molecule_geometry) #is the number of atoms in the molecule

                arguments = (N, eta, alpha, gamma1, gamma2, zeta_max_i, phi_max, J)

                # generate values for errors epsilon_PEA, epsilon_HS, epsilon_S, eps_H, eps_taylor
                for _ in range(self.runs):
                    optimized_errors = self.calculate_optimized_errors(5, methods_taylor.configuration_interaction, arguments)

                    # alpha, gamma1, gamma2 are used to calculate K0, K1, K2 (see eq D14 in overleaf)
                    self.costs['configuration_interaction'] += [methods_taylor.configuration_interaction(
                        optimized_errors.x,
                        N,
                        eta,    
                        alpha,
                        gamma1,
                        gamma2,
                        zeta_max_i,
                        phi_max,
                        J)]

        
        elif method == 'low_depth_trotter' or method == 'low_depth_taylor' or method == 'low_depth_taylor_on_the_fly':
            methods_plane_waves = plane_waves_methods.Plane_waves_methods(self.tools)

            # This methods are plane waves, so instead of calling self.molecule.get_basic_parameters() one should call self.molecule.build_grid()
            # grid_length is the only parameter of build_grid. Should be calculated such that the number of basis functions
            #   is ~= 100*self.molecule_data.n_orbitals*2. grid_length ~= int(np.cbrt(100*self.molecule.molecule_data.n_orbitals * 2))
            # Omega is returned by self.molecule.build_grid()
            # J = len(self.molecule.geometry) #is the number of atoms in the molecule

            if method == 'low_depth_trotter':

                grid_length = int(round((self.molecule.N * 100) ** (1/3)))
                if not hasattr(self.molecule, 'eta') or not hasattr(self.molecule, 'Omega') or not hasattr(self.molecule, 'N_grid'):
                    grid = self.molecule.build_grid(grid_length)

                N_grid = self.molecule.N_grid
                eta = self.molecule.eta
                Omega = self.molecule.Omega

                arguments = (N_grid, eta, Omega)

                # generate values for errors epsilon_PEA, epsilon_HS, epsilon_S
                for _ in range(self.runs):
                    optimized_errors = self.calculate_optimized_errors(3, methods_plane_waves.low_depth_trotter, arguments)

                    self.costs['low_depth_trotter'] += [methods_plane_waves.low_depth_trotter(
                        optimized_errors.x,
                        N_grid, 
                        eta, 
                        Omega)]

            elif method == 'low_depth_taylor':

                grid_length = int(round((self.molecule.N * 100) ** (1/3)))
                if not hasattr(self.molecule, 'lambda_value_grid') or not hasattr(self.molecule, 'Lambda_value_grid') or not hasattr(self.molecule, 'N_grid'):
                    grid = self.molecule.build_grid(grid_length)

                N_grid = self.molecule.N_grid
                lambda_value_grid  = self.molecule.lambda_value_grid 
                Lambda_value_grid  = self.molecule.Lambda_value_grid 
                H_norm_lambda_ratio = self.tools.config_variables['h_norm_lambda_ratio']

                arguments = (N_grid, lambda_value_grid, Lambda_value_grid, H_norm_lambda_ratio)

                # generate value for errors epsilon_PEA, epsilon_HS, epsilon_S
                for _ in range(self.runs):
                    optimized_errors = self.calculate_optimized_errors(3, methods_plane_waves.low_depth_taylor, arguments)

                    self.costs['low_depth_taylor'] += [methods_plane_waves.low_depth_taylor(
                        optimized_errors.x,
                        N_grid, 
                        lambda_value_grid, 
                        Lambda_value_grid, 
                        H_norm_lambda_ratio)]

            elif method == 'low_depth_taylor_on_the_fly':

                grid_length = int(round((self.molecule.N * 100) ** (1/3)))
                if not hasattr(self.molecule, 'lambda_value_grid') or not hasattr(self.molecule, 'Omega') or not hasattr(self.molecule, 'Gamma_grid') or not hasattr(self.molecule, 'eta') or not hasattr(self.molecule, 'N_grid'):
                    grid = self.molecule.build_grid(grid_length)

                N_grid = self.molecule.N_grid
                eta = self.molecule.eta
                Gamma_grid = self.molecule.Gamma_grid 
                lambda_value_grid = self.molecule.lambda_value_grid 
                Omega = self.molecule.Omega
                
                x_max = self.molecule.xmax
                J = len(self.molecule.molecule_geometry) #is the number of atoms in the molecule

                arguments = (N_grid, eta, Gamma_grid, lambda_value_grid, Omega, J, x_max)

                # generate value for errors epsilon_PEA, epsilon_HS, epsilon_S, epsilon_H, epsilon_tay
                for _ in range(self.runs):
                    optimized_errors = self.calculate_optimized_errors(5, methods_plane_waves.low_depth_taylor_on_the_fly, arguments)

                    # find x_max from cell volume assuming a perfect cube centered on 0
                    self.costs['low_depth_taylor_on_the_fly'] += [methods_plane_waves.low_depth_taylor_on_the_fly(
                        optimized_errors.x,
                        N_grid, 
                        eta,
                        Gamma_grid,
                        lambda_value_grid , 
                        Omega,
                        J, 
                        x_max)]

        elif method == 'linear_t' or method == 'sparsity_low_rank':

            methods_qrom = qrom_methods.QROM_methods(self.tools)

            if method == 'linear_t':

                grid_length = int(round((self.molecule.N * 100) ** (1/3)))
                if not hasattr(self.molecule, 'lambda_value_grid') or not hasattr(self.molecule, 'N_grid'):
                    grid = self.molecule.build_grid(grid_length)

                N_grid = self.molecule.N_grid
                lambda_value_grid = self.molecule.lambda_value_grid
                H_norm_lambda_ratio = self.tools.config_variables['h_norm_lambda_ratio']

                arguments = (N_grid, lambda_value_grid, H_norm_lambda_ratio)

                # generate value for errors epsilon_PEA, epsilon_S
                for _ in range(self.runs):
                    optimized_errors = self.calculate_optimized_errors(2, methods_qrom.linear_T, arguments)
                    
                    self.costs['linear_t'] += [methods_qrom.linear_T(
                        optimized_errors.x,
                        N_grid, 
                        lambda_value_grid ,
                        H_norm_lambda_ratio)]

            elif method == 'sparsity_low_rank':

                if not hasattr(self.molecule, 'sparsity_d') or not hasattr(self.molecule, 'final_rank') or not hasattr(self.molecule, 'lambda_value_low_rank'):
                    self.molecule.low_rank_approximation(sparsify = True)

                N = self.molecule.N
                lambda_value = self.molecule.lambda_value_low_rank
                sparsity_d = self.molecule.sparsity_d 
                final_rank = self.molecule.final_rank
                H_norm_lambda_ratio = self.tools.config_variables['h_norm_lambda_ratio']

                arguments = (N, lambda_value, final_rank, H_norm_lambda_ratio, sparsity_d)
                
                # generate value for errors epsilon_PEA, epsilon_S
                for _ in range(self.runs):
                    optimized_errors = self.calculate_optimized_errors(2, methods_qrom.sparsity_low_rank, arguments)

                    self.costs['sparsity_low_rank'] += [methods_qrom.sparsity_low_rank(
                        optimized_errors.x,
                        N, 
                        lambda_value,
                        final_rank, 
                        H_norm_lambda_ratio,
                        sparsity_d)]
        
        elif method == 'interaction_picture' or method == 'sublinear_scaling':

            methods_interaction_picture = interaction_picture.Interaction_picture(self.tools)

            if method == 'interaction_picture':

                grid_length = int(round((self.molecule.N * 100) ** (1/3)))
                
                if not hasattr(self.molecule, 'lambda_value_T') or not hasattr(self.molecule, 'lambda_value_U_V') or not hasattr(self.molecule, 'Gamma_grid') or not hasattr(self.molecule, 'N_grid'):
                    grid = self.molecule.build_grid(grid_length)
                    self.molecule.lambda_of_Hamiltonian_terms_2nd(grid)

                lambda_value_T = self.molecule.lambda_value_T 
                lambda_value_U_V = self.molecule.lambda_value_U_V

                N_grid = self.molecule.N_grid
                Gamma_grid  = self.molecule.Gamma_grid 

                arguments = (N_grid, Gamma_grid, lambda_value_T, lambda_value_U_V)

                # generate value for errors epsilon_S, epsilon_HS, epsilon_PEA
                for _ in range(self.runs):
                    optimized_errors = self.calculate_optimized_errors(3, methods_interaction_picture.interaction_picture, arguments)

                    self.costs['interaction_picture'] += [methods_interaction_picture.interaction_picture(
                        optimized_errors.x,
                        N_grid, 
                        Gamma_grid, 
                        lambda_value_T, 
                        lambda_value_U_V)]
            
            
            # TO BE DELETED
            elif method == 'sublinear_scaling':

                grid_length = int(round((self.molecule.N * 100) ** (1/3)))
                if not hasattr(self.molecule, 'lambda_value_T') or not hasattr(self.molecule, 'lambda_value_U_V') or not hasattr(self.molecule, 'Gamma_grid') or not hasattr(self.molecule, 'N_grid'):
                    grid = self.molecule.build_grid(grid_length)

                N_grid = self.molecule.N_grid
                eta = self.molecule.eta
                Gamma_grid  = self.molecule.Gamma_grid 
                Omega = self.molecule.Omega
                

                J = len(self.molecule.molecule_geometry) #is the number of atoms in the molecule

                Omega = self.molecule.Omega
                self.molecule.lambda_of_Hamiltonian_terms_1st(eta, Omega, N_grid)
                lambda_value_T, lambda_value_U_V = self.molecule.lambda_value_T, self.molecule.lambda_value_U_V

                arguments = (N_grid, eta, Gamma_grid , lambda_value_T, lambda_value_U_V, J)

                # generate value for errors epsilon_S, epsilon_HS, epsilon_PEA, epsilon_mu, epsilon_M_0, epsilon_R
                for _ in range(self.runs):
                    optimized_errors = self.calculate_optimized_errors(6, methods_interaction_picture.sublinear_scaling_interaction, arguments)

                    self.costs['sublinear_scaling'] += [methods_interaction_picture.sublinear_scaling_interaction(
                        optimized_errors.x,
                        N_grid, 
                        eta, 
                        Gamma_grid, 
                        lambda_value_T, 
                        lambda_value_U_V,
                        J)]

        else:
            print('<*> ERROR: method', method, 'not implemented or not existing')

        if self.molecule_info_type == 'name':
            json_name = str(self.molecule.molecule_info)+ '_' +  str(self.basis)
            self.molecule.save(json_name = 'parameters/'+json_name)

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


    def calculate_time(self, T_gates, p_fail = 1e-1, p_surface_step = 1e-3, P_inject = 5e-3, P_threshold = 5.7e-3, t_cycle = 2e-7, AAA_factories = 1e3):
        '''
        Calculates the time required to synthesise the T_gates.
        Based on Appendix M from PHYSICAL REVIEW A 86, 032324 (2012); "Surface codes: Towards practical large-scale quantum computation" by Austin G. Fowler

        Arguments:
        T_gates: int; the numer of T gates that we have to synthesise
        p_fail: int;  the probability of failure.
        P_inject: float; the failure probability in injected states
        P_threshold: float; the surface code failure probability
        t_cycle: float; the time of one cycle of the surface code
        AAA_factories: float; the number of AAA factories available working in parallel

        Returns:
        time: float; the time (seconds) required to synthesise the T_gates
        '''

        P_A = p_fail/T_gates

        p_list = [P_inject]
        assert(35*P_inject**3 < 1)
        while p_list[-1] < P_A:
            p = 35*p_list[-1]**3
            p_list.append(p)

        def distance_2_error(distance,ord):
            de = np.floor((int(distance)+1)/2)
            PL = 3e-2*(p_surface_step/P_threshold)**de
            p_i = 15**ord*16*3*2*1.25*distance*PL
            return p_i

        vfunc = np.vectorize(distance_2_error)

        constraints = NonlinearConstraint(fun = lambda distances: vfunc(distances, list(range(len(p_list)))), lb = -np.inf, ub = p_list)

        x0 = [17]
        for i in range(len(p_list)-1):
            x0.append(x0[-1]*2)
            
        res = minimize(fun = lambda distances: distances.sum(), x0 = x0, method = 'SLSQP', constraints = constraints)
        distances = res.x

        distances = [int(d) for d in distances]

        code_cycles = 8*1.25*sum(distances)

        time = code_cycles*t_cycle*T_gates/AAA_factories

        return time