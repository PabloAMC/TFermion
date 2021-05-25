import utils
from molecule import Molecule
import numpy as np
import cost_calculator

#Read config file with the QFold configuration variables
config_path = './config/config.json'
tools = utils.Utils(config_path)

args = tools.parse_arguments()
try:
    charge = args.charge
except:
    charge = 0

# Some easy examples: HYDROFLUORIC ACID, Ammonia, water, methane, O2, CO2, O3, NaCl
# More complex examples (beware the time): https://pubchem.ncbi.nlm.nih.gov/compound/7966 (cyclohexanol, MA's choosing)
# https://pubchem.ncbi.nlm.nih.gov/compound/25000034 (Tetrahedral, 1.66 Amstrong), https://pubchem.ncbi.nlm.nih.gov/compound/167316 (table 1 in https://chemistry-europe.onlinelibrary.wiley.com/doi/full/10.1002/cphc.200700128?casa_token=fYXpuPMymU4AAAAA%3Ao0dz2LXXn8yVq56nOt5ZrV92HiuzItsffXm6Nn_O9z3hXt2d2Sm2qVX-GZwQsnQ_z4PPPPrN2jSqfIg)
#molecule = Molecule(name = 'H', tools = tools)
molecule = Molecule(name = args.molecule_name, tools = tools, charge = charge)

#molecule.low_rank_approximation(occupied_indices = [0,1,2], active_indices = [3,4], virtual_indices = [5,6], sparsify = True)
#ne_act_cas, n_mocore, n_mocas, n_movir = molecule.active_space(ao_labels=['O 2pz'])

c_calculator = cost_calculator.Cost_calculator(molecule, tools)
c_calculator.calculate_cost(args.method)
print('The cost to calculate the energy of', args.molecule_name,'with method', args.method, 'is', "{:e}".format(c_calculator.costs[args.method]))

#### TESTS ###

# QDRIFT: python3 main.py water qdrift 'C 2p' FAIL
# RAND-HAM: python3 main.py water rand_ham 'C 2p' FAIL 
# Taylor naive: python3 main.py water taylor_naive 'C 2p' FAIL
# Taylor on the fly: python3 main.py water taylor_on_the_fly 'C 2p' FAIL
# Configuration interaction: python3 main.py water configuration_interaction 'C 2p' FAIL
# Low Depth Trotter: python3 main.py water low_depth_trotter 'C 2p' FAIL
# Low Depth Taylor: python3 main.py water low_depth_taylor 'C 2p' FAIL
# Low Depth On The Fly: python3 main.py water low_depth_on_the_fly 'C 2p' FAIL
# Linear T: python3 main.py linear_T 'C 2p' FAIL
# Sparsity Low Rank: python3 main.py sparsity_low_rank 'C 2p' FAIL
# Interaction Picture: python3 main.py interaction_picture 'C 2p' FAIL
# Sublinear Scaling: python3 main.py sublinear_scaling 'C 2p' FAIL