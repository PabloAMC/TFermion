How to contribute (new method)
=============

| Type  | Method | Description |
|:-------------: |:-------------: |:-------------: |
| Add code example  | SHC Trotter | Code necessary to add a new method |

`Guide of how to contribute to TFermion adding the cost estimation of a new method in 7 steps`

Code (example to add method SHC Trotter)
-------------

1. Add method name to TFermion

`Add method name in class utils.py tp the variable METHODS`

```
METHODS = [
        'qdrift', 
        'rand_ham', 
        'taylor_naive', 
        'taylor_on_the_fly', 
        'configuration_interaction',
        'low_depth_trotter', 
        'low_depth_taylor', 
        'low_depth_taylor_on_the_fly', 
        'linear_t', 
        'sparsity_low_rank', 
        'interaction_picture',
        'shc_trotter']
```

2. Add method to the index of available method

`Add an option to execute the new method in cost_calculator.py (it can be included in the same family of other algorithms or in a new family, in this case it is included in the same family of low_depth_trotter)`

```
elif method == 'low_depth_trotter' or method == 'shc_trotter':

  if method == 'shc_trotter':
```

3. Check if necessary parameters are already calculated

`Check if in the parameters for the selected molecule have been already calculated and caculate them if they are not. The calculated parameters of the molecules are in a file stored in parameters folder`

```
if not hasattr(self.molecule, 'eta') or not hasattr(self.molecule, 'Omega') or not hasattr(self.molecule, 'N_grid'):
  grid = self.molecule.build_grid(grid_length)
```

4. Compute the necessary parameters

`Compute the parameters necessaries to execute the new method and save them in the arguments list`

```
N_grid = self.molecule.N_grid
eta = self.molecule.eta
Omega = self.molecule.Omega

arguments = (self.p_fail, N_grid, eta, Omega)
```

5. Declare the errors

`Declare and calculate the optimized errors necessary to execute the new method. It just necessary to indicate the number of errors, the method and the arguments`

```
optimized_errors = self.calculate_optimized_errors(3, methods_plane_waves.shc_trotter, arguments)
```

6. Execute the method and save the result

`Execute the method with the optimized errors and the arguments. Then save the result in the data structure that accumulates the calculated costs`

```
self.costs['shc_trotter'] += [methods_plane_waves.shc_trotter(
                        optimized_errors.x,
                        self.p_fail,
                        N_grid, 
                        eta, 
                        Omega)]
```
7. Add the code of the new energy calculation mode

`Add the code in the same class than other methods for same family or create a new class (new family). In this case, this new method is included in plane_waves_methods.py`

```
# Similar to low_depth_trotter but with tighter SHC bounds for the commutator obtained from https://journals.aps.org/pra/abstract/10.1103/PhysRevA.105.012403
def shc_trotter(self,epsilons, p_fail, N, eta, Omega):

    epsilon_QPE = epsilons[0]
    epsilon_HS = epsilons[1]
    epsilon_S = epsilons[2]

    t = np.pi/(2*epsilon_QPE)*(1/2+1/(2*p_fail))
    max_U_V = (Omega**(1/3)*eta)/np.pi 
    nu_max = np.sqrt(3*(N**(1/3))**2)
    norm_T = 2*np.pi**2*eta/(Omega**(2/3))* nu_max**2

    TVT_commutator = 4*norm_T**2*max_U_V*eta*(4*eta+1)
    TVV_commutator = 12*norm_T*max_U_V**2*eta**2*(2*eta+1)
    W2= (TVT_commutator + TVV_commutator)/12

    r = np.sqrt(t**3/epsilon_HS * W2)

    # Arbitrary precision rotations, does not include the Ry gates in F_2
    single_qubit_rotations = r*(8*N + 2*8*N*(8*N-1) + 8*N + N*np.ceil(np.log2(N/2))) # U, V, T and FFFT single rotations; the 2 comes from the controlled rotations, see appendix A
    epsilon_SS = epsilon_S/single_qubit_rotations

    exp_UV_cost = 8*N*(8*N-1)*self.tools.c_pauli_rotation_synthesis(epsilon_SS) + 8*N*self.tools.pauli_rotation_synthesis(epsilon_SS)
    exp_T_cost = 8*N*self.tools.pauli_rotation_synthesis(epsilon_SS)
    F2 = 2
    FFFT_cost = N/2*np.ceil(np.log2(N))*F2 + N/2*(np.ceil(np.log2(N))-1)*self.tools.pauli_rotation_synthesis(epsilon_SS) 

    return r*(2*exp_UV_cost + exp_T_cost + 2*FFFT_cost)
```


Finally, the whole code in cost_calculator.py can help to understand this modular approach:

```
elif method == 'low_depth_trotter' or method == 'low_depth_taylor' or method == 'low_depth_taylor_on_the_fly' or method == 'shc_trotter':
  methods_plane_waves = plane_waves_methods.Plane_waves_methods(self.tools)

  # This methods are plane waves, so instead of calling self.molecule.get_basic_parameters() one should call self.molecule.build_grid()
  # grid_length is the only parameter of build_grid. Should be calculated such that the number of basis functions
  #   is ~= 100*self.molecule_data.n_orbitals. grid_length ~= int(np.cbrt(100*self.molecule.molecule_data.n_orbitals * 2))
  # Omega is returned by self.molecule.build_grid()
  # J = len(self.molecule.geometry) #is the number of atoms in the molecule

  if method == 'low_depth_trotter':

      grid_length = int(round((self.molecule.N * self.tools.config_variables['gauss2plane_overhead']) ** (1/3)))
      if not hasattr(self.molecule, 'eta') or not hasattr(self.molecule, 'Omega') or not hasattr(self.molecule, 'N_grid'):
          grid = self.molecule.build_grid(grid_length)

      N_grid = self.molecule.N_grid
      eta = self.molecule.eta
      Omega = self.molecule.Omega

      arguments = (self.p_fail, N_grid, eta, Omega)

      # generate values for errors epsilon_PEA, epsilon_HS, epsilon_S
      for _ in range(self.runs):
          optimized_errors = self.calculate_optimized_errors(3, methods_plane_waves.low_depth_trotter, arguments)

          self.costs['low_depth_trotter'] += [methods_plane_waves.low_depth_trotter(
              optimized_errors.x,
              self.p_fail,
              N_grid, 
              eta, 
              Omega)]

  elif method == 'shc_trotter':

      grid_length = int(round((self.molecule.N * self.tools.config_variables['gauss2plane_overhead']) ** (1/3)))
      if not hasattr(self.molecule, 'eta') or not hasattr(self.molecule, 'Omega') or not hasattr(self.molecule, 'N_grid'):
          grid = self.molecule.build_grid(grid_length)

      N_grid = self.molecule.N_grid
      eta = self.molecule.eta
      Omega = self.molecule.Omega

      arguments = (self.p_fail, N_grid, eta, Omega)

      # generate values for errors epsilon_PEA, epsilon_HS, epsilon_S
      for _ in range(self.runs):
          optimized_errors = self.calculate_optimized_errors(3, methods_plane_waves.shc_trotter, arguments)

          self.costs['shc_trotter'] += [methods_plane_waves.shc_trotter(
              optimized_errors.x,
              self.p_fail,
              N_grid, 
              eta, 
              Omega)]
```
