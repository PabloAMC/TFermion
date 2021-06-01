import utils
from molecule import Molecule
import cost_calculator
import datetime
import time

print('\n###################################################################')
print('##                             QPHASE                            ##')
print('##                                                               ##')
print('##      We will see what it is that (Not Google paper copy)      ##')
print('###################################################################\n')

start_time = time.time()

#Read config file with the QFold configuration variables
config_path = './config/config.json'
tools = utils.Utils(config_path)

args = tools.parse_arguments()
if args.charge == None: args.charge = 0

# Some easy examples: HYDROFLUORIC ACID, Ammonia, water, methane, O2, CO2, O3, NaCl
# More complex examples (beware the time): https://pubchem.ncbi.nlm.nih.gov/compound/7966 (cyclohexanol, MA's choosing)
# https://pubchem.ncbi.nlm.nih.gov/compound/25000034 (Tetrahedral, 1.66 Amstrong), https://pubchem.ncbi.nlm.nih.gov/compound/167316 (table 1 in https://chemistry-europe.onlinelibrary.wiley.com/doi/full/10.1002/cphc.200700128?casa_token=fYXpuPMymU4AAAAA%3Ao0dz2LXXn8yVq56nOt5ZrV92HiuzItsffXm6Nn_O9z3hXt2d2Sm2qVX-GZwQsnQ_z4PPPPrN2jSqfIg)
#molecule = Molecule(name = 'H', tools = tools)
molecule = Molecule(name = args.molecule_name, tools = tools, charge = args.charge)

#molecule.low_rank_approximation(occupied_indices = [0,1,2], active_indices = [3,4], virtual_indices = [5,6], sparsify = True)
#ne_act_cas, n_mocore, n_mocas, n_movir = molecule.active_space(ao_labels=['O 2pz'])

c_calculator = cost_calculator.Cost_calculator(molecule, tools)
c_calculator.calculate_cost(args.method)
print('The cost to calculate the energy of', args.molecule_name,'with method', args.method, 'is', "{:e}".format(c_calculator.costs[args.method]))

execution_time = time.time() - start_time

print('\n** -------------------------------------------------- **')
print('**                                                    **')
print('** Execution time     =>', str(datetime.timedelta(seconds=execution_time)) ,' in hh:mm:ss  **')
print('********************************************************\n\n')


#### TESTS ###

# QDRIFT: python3 main.py water qdrift 'C 2p' OK
# RAND-HAM: python3 main.py water rand_ham 'C 2p' OK 
# Taylor naive: python3 main.py water taylor_naive 'C 2p' OK
# Taylor on the fly: python3 main.py water taylor_on_the_fly 'C 2p' OK
# Configuration interaction: python3 main.py water configuration_interaction 'C 2p' OK
# Low Depth Trotter: python3 main.py water low_depth_trotter 'C 2p' OK
# Low Depth Taylor: python3 main.py water low_depth_taylor 'C 2p' OK
# Low Depth On The Fly: python3 main.py water low_depth_taylor_on_the_fly 'C 2p' OK
# Linear T: python3 main.py water linear_t 'C 2p' OK
# Sparsity Low Rank: python3 main.py water sparsity_low_rank 'C 2p' OK
# Interaction Picture: python3 main.py water interaction_picture 'C 2p' FAIL
# Sublinear Scaling: python3 main.py water sublinear_scaling 'C 2p' OK