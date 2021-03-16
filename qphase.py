import utils
import molecule
import cost_calculator

import numpy as np
from itertools import combinations
import scipy
from scipy.optimize import minimize
from scipy.special import binom, gamma
from scipy.integrate import quad, dblquad
from scipy import integrate
import sympy
import math

from openfermion.chem import geometry_from_pubchem, MolecularData
from openfermionpsi4 import run_psi4
from  openfermion.transforms  import  get_fermion_operator,  jordan_wigner
import openfermion
from openfermion.utils import Grid
from openfermion.hamiltonians import plane_wave_external_potential, plane_wave_potential, plane_wave_kinetic
from openfermion.hamiltonians import plane_wave_hamiltonian
from openfermion.hamiltonians import dual_basis_external_potential, dual_basis_potential, dual_basis_kinetic

## The aim of this notebook is to calculate a cost estimation of different methods to calculate the energy of a system with different Phase Estimation protocols
## IMPORTANT: to these cost, we have to add the QFT cost, which is minor, and has an associated error.

# Finding the molecule parameters
# Docs https://quantumai.google/reference/python/openfermion/

#Read config file with the QFold configuration variables
config_path = './config/config.json'
tools = utils.Utils(config_path)

args = tools.parse_arguments()

print('#################################################################################')
print('##                                    QPHASE                                   ##')
print('##                                                                             ##')
print('## Tool to calculate the T gates cost of any molecule energy calculator method ##')
print('#################################################################################\n')

molecule = molecule.Molecule(args.molecule_name, tools)

c_calculator = cost_calculator.Cost_calculator(molecule, tools)
c_calculator.calculate_cost(args.method)
print('The cost to calculate the energy of', args.molecule_name,'with method', args.method, 'is', c_calculator.costs[args.method])