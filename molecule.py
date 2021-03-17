import openfermion

from openfermionpsi4 import run_psi4

from openfermion.utils import Grid
from openfermion.chem import geometry_from_pubchem, MolecularData
from openfermion.hamiltonians import plane_wave_hamiltonian
from openfermion.transforms  import  get_fermion_operator
from openfermion.transforms import jordan_wigner

import numpy as np

class Molecule:

    def __init__(self, name, tools):

        self.molecule_name = name
        self.tools = tools

        self.gamma_threshold = self.tools.config_variables['gamma_threshold']

        self.molecule_geometry = geometry_from_pubchem(self.molecule_name)
        self.molecule_data = MolecularData(self.molecule_geometry, self.tools.config_variables['basis'], multiplicity = 1)
        self.molecule_psi4 = run_psi4(self.molecule_data,run_scf=True, run_mp2=True, run_fci=False)

        self.lambda_value, self.Lambda_value, self.N, self.gamma = self.get_parameters(self.molecule_psi4)

    def get_parameters(self, molecule):

        lambda_value_one_body, Lambda_value_one_body, N_one_body, gamma_one_body = self.get_parameters_from_row(self.molecule_psi4.one_body_integrals, 0, -1, 0, 0)
        lambda_value_two_body, Lambda_value_two_body, N_two_body, gamma_two_body = self.get_parameters_from_row(self.molecule_psi4.two_body_integrals, 0, -1, 0, 0)

        assert N_one_body == N_two_body

        return lambda_value_one_body+lambda_value_two_body, max(Lambda_value_one_body, Lambda_value_two_body), N_one_body, gamma_one_body+gamma_two_body

    def build_grid(self):
        grid = Grid(dimensions = 3, length = 5, scale = 1.) # La complejidad
        plane_wave_H = plane_wave_hamiltonian(grid, self.molecule_geometry, True)

    # recursive method that iterates over all rows of a molecule to get the parameters:
    # lambda_value is the sum all coefficients of the hamiltonian (sum of all terms)
    # Lambda_value is the maximum value of all terms
    # N is the number of orbitals
    # gamma is the total number of elements (without counting values under some threshold)
    def get_parameters_from_row(self, row, lambda_value, Lambda_value, N, gamma):

        for r in row:

            if type(r) == np.ndarray:
                lambda_value, Lambda_value, N, gamma = self.get_parameters_from_row(r, lambda_value, Lambda_value, N, gamma)
            else:

                # lambda_value is the sum of all terms of all rows
                lambda_value += abs(r)
                
                # Lambda_value is the maximum of all terms of all rows
                if abs(r) > Lambda_value: Lambda_value = abs(r)

                # N is the number of orbitals and it should be equal in all rows (if it is 0, initialize it)
                if N == 0: N = len(row) 
                else: assert N == len(row)

                # gamma is the count of elements above a threshold (threshold close to 0 or 0 to delete small terms)
                if abs(r) > self.gamma_threshold: gamma += 1

        return lambda_value, Lambda_value, N, gamma
