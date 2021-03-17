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

        self.molecule_geometry = geometry_from_pubchem(self.molecule_name)
        self.molecule_data = MolecularData(self.molecule_geometry, self.tools.config_variables['basis'], multiplicity = 1)
        self.molecule_psi4 = run_psi4(self.molecule_data,run_scf=True, run_mp2=True, run_fci=False)

    '''
    To obtain these Hamiltonians one must choose to study the system without a spin degree of freedom (spinless),
    one must the specify dimension in which the calculation is performed (n_dimensions, usually 3),
    one must specify how many plane waves are in each dimension (grid_length)
    and one must specify the length scale of the plane wave harmonics in each dimension (length_scale)
    and also the locations and charges of the nuclei.

    Taken from https://quantumai.google/openfermion/tutorials/intro_to_openfermion
    '''
    def get_lambda_from_hamiltonian(self):

        one_body_value = sum(sum(abs(self.molecule_psi4.one_body_integrals)))

        two_body_value = sum(sum(sum(sum(abs(self.molecule_psi4.two_body_integrals)))))

        return one_body_value + two_body_value

    def get_orbitals(self):
        return len(self.molecule_psi4.one_body_integrals[0])

    def build_grid(self):
        grid = Grid(dimensions = 3, length = 5, scale = 1.) # La complejidad
        plane_wave_H = plane_wave_hamiltonian(grid, self.molecule_geometry, True)