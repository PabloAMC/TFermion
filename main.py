import utils
from molecule import Molecule
import numpy as np
import cost_calculator

#Read config file with the QFold configuration variables
config_path = './config/config.json'
tools = utils.Utils(config_path)

args = tools.parse_arguments()

molecule = Molecule(name = args.molecule_name, tools = tools)

#molecule.low_rank_approximation(occupied_indices = [0,1,2], active_indices = [3,4], virtual_indices = [5,6], sparsify = True)
#ne_act_cas, n_mocore, n_mocas, n_movir = molecule.active_space(ao_labels=['O 2pz'])

c_calculator = cost_calculator.Cost_calculator(molecule, tools)
c_calculator.calculate_cost(args.method)
print('The cost to calculate the energy of', args.molecule_name,'with method', args.method, 'is', "{:e}".format(c_calculator.costs[args.method]))