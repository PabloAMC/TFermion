import utils
import cost_calculator
import datetime
import time
import pandas as pd

from molecule import Molecule
from molecule import Molecule_Hamiltonian
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

print('\n################################################################################################')
print('##                                          T-FERMION                                         ##')
print('##                                                                                            ##')
print('## Tool to calculate the T gate cost of a quantum method for energy calculation of a molecule ##')
print('################################################################################################\n')

start_time = time.time()

#Read config file with the QFold configuration variables
config_path = './config/config.json'
tools = utils.Utils(config_path)

args = tools.parse_arguments()
if args.charge == None: args.charge = 0

# Basis sets that we could be using: STO-3G, DZ, 6-311G, cc-pVDZ, and cc-pVTZ basis sets, in increasing precision order.

# Some easy examples: HYDROFLUORIC ACID, Ammonia, water, methane, O2, CO2, O3, NaCl
# More complex examples (beware the time): https://pubchem.ncbi.nlm.nih.gov/compound/7966 (cyclohexanol, MA's choosing)
# https://pubchem.ncbi.nlm.nih.gov/compound/25000034 (Tetrahedral, 1.66 Amstrong), https://pubchem.ncbi.nlm.nih.gov/compound/167316 (table 1 in https://chemistry-europe.onlinelibrary.wiley.com/doi/full/10.1002/cphc.200700128?casa_token=fYXpuPMymU4AAAAA%3Ao0dz2LXXn8yVq56nOt5ZrV92HiuzItsffXm6Nn_O9z3hXt2d2Sm2qVX-GZwQsnQ_z4PPPPrN2jSqfIg)
#molecule = Molecule(name = 'H', tools = tools)

molecule_info_type = tools.check_molecule_info(args.molecule_info)

if not molecule_info_type:
    molecule_info_type = 'name'
    dictionary = {}
    methods = ['qdrift', 'rand_ham', 'taylor_naive', 'taylor_on_the_fly', 'configuration_interaction',
                'low_depth_trotter', 'low_depth_taylor', 'low_depth_taylor_on_the_fly', 
                'linear_t', 'sparsity_low_rank', 'interaction_picture']

    for molecule_info in tools.config_variables['molecules']:
        args.molecule_info = molecule_info
        dictionary[molecule_info] = {}

        molecule = None
        if molecule_info_type == 'name' or molecule_info_type == 'geometry':
            molecule = Molecule(molecule_info = args.molecule_info, molecule_info_type = molecule_info_type, tools = tools, charge = args.charge)
        elif molecule_info_type == 'hamiltonian':
            molecule = Molecule_Hamiltonian(molecule_info = args.molecule_info, tools = tools)
        else: # if there is no match between the input file extension and the requiered, finish the program
            exit()

        #Active space
        if args.ao_labels:
            ne_act_cas = molecule.active_space(args.ao_labels[0].replace('\\',''))

        #molecule.low_rank_approximation(occupied_indices = [0,1,2], active_indices = [3,4], virtual_indices = [5,6], sparsify = True)
        #ne_act_cas, n_mocore, n_mocas, n_movir = molecule.active_space(ao_labels=['O 2pz'])

        c_calculator = cost_calculator.Cost_calculator(molecule, tools, molecule_info_type)
        for method in methods:
            c_calculator.calculate_cost(method)
            c_calculator.costs[method] = [x for x in c_calculator.costs[method] if (not np.isnan(x) and not np.isinf(x))]
            print(method, molecule_info, len(c_calculator.costs[method]))
            median = np.nanmedian(c_calculator.costs[method])
            dictionary[molecule_info][method] = "{:0.2e}".format(median)

    pd.DataFrame(dictionary).to_csv('./results/results.csv')


else: 
    molecule = None
    if molecule_info_type == 'name' or molecule_info_type == 'geometry':
        molecule = Molecule(molecule_info = args.molecule_info, molecule_info_type = molecule_info_type, tools = tools, charge = args.charge)
    elif molecule_info_type == 'hamiltonian':
        molecule = Molecule_Hamiltonian(molecule_info = args.molecule_info, tools = tools)
    else: # if there is no match between the input file extension and the requiered, finish the program
        exit()

    #Active space
    if args.ao_labels:
        ne_act_cas = molecule.active_space(args.ao_labels[0].replace('\\',''))

    #molecule.low_rank_approximation(occupied_indices = [0,1,2], active_indices = [3,4], virtual_indices = [5,6], sparsify = True)
    #ne_act_cas, n_mocore, n_mocas, n_movir = molecule.active_space(ao_labels=['O 2pz'])

    c_calculator = cost_calculator.Cost_calculator(molecule, tools)
    c_calculator.calculate_cost(args.method)

    cost_module = 'total'

    '''
    p = figure(
        #title='Evolution of tts with different steps', # Usually graphs do not have title
        x_axis_type="log",
        y_axis_type="log",
        x_range= (10**2, 10**9),
        y_range= (10**11, 10**16),
        plot_height=700,
        plot_width=800)
    '''

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))

    ax.set_xlim([10**2, 10**9])
    ax.set_ylim([10**15, 10**18])
    ax.set_yscale("log")
    ax.set_xscale("log")

    points = c_calculator.costs[args.method]

    counter = -1
    for chemical_acc in points:

        counter+=1

        for cost_object in chemical_acc:
        
            x_value = cost_object[0]

            if cost_module == 'HF': median = np.nanmedian([element[0] for element in cost_object[1]])
            elif cost_module == 'QPE': median = np.nanmedian([element[2] for element in cost_object[1]])
            elif cost_module == 'total': median = np.nanmedian([sum(element) for element in cost_object[1]])
        
            if counter == 0:

                # add a line renderer
                ax.scatter(x_value, median, marker="h", c="blue", alpha=0.5, s=200, label=r'$\epsilon_{PEA}=0.014eV$')

            elif counter == 1:

                # add a line renderer
                ax.scatter(x_value, median, marker="s", c="green", alpha=0.5, s=200, label=r'$\epsilon_{PEA}=0.043eV$')

            if counter == 2:

                # add a line renderer
                ax.scatter(x_value, median, marker="*", c="orange", alpha=0.5, s=200, label=r'$\epsilon_{PEA}=0.129eV$')

            if counter == 3:

                # add a line renderer
                ax.scatter(x_value, median, marker="d", c="red", alpha=0.5, s=200, label=r'$\epsilon_{PEA}=0.387eV$')

    ax.set_ylabel(r'Toffoli gate cost', fontsize=20)
    ax.set_xlabel(r'Number of plane waves, $N$', fontsize=20)

    # First plot: two legend keys for a single entry
    p1 = ax.scatter([0], [0], c='blue', marker='h', alpha=0.5, s=100)
    p2 = ax.scatter([0], [0], c='green', marker='s', alpha=0.5, s=100)
    p3 = ax.scatter([0], [0], c='orange', marker='*', alpha=0.5, s=100)
    p4 = ax.scatter([0], [0], c='red', marker='d', alpha=0.5, s=100)

    # Assign two of the handles to the same legend entry by putting them in a tuple
    # and using a generic handler map (which would be used for any additional
    # tuples of handles like (p1, p3)).
    l = ax.legend([p1, p2, p3, p4], [r'$\epsilon_{QPE}=0.014eV$', r'$\epsilon_{QPE}=0.043eV$', r'$\epsilon_{QPE}=0.129eV$', r'$\epsilon_{QPE}=0.387eV$'], scatterpoints=1,
                numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)}, loc='upper left', borderpad=1, prop={'size': 12}, labelspacing=1)

    ax.tick_params(axis='x')

    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)

    plt.savefig('plot.pdf')  

    print('The cost to calculate the energy of', args.molecule_info,'with method', args.method, 'is', "{:0.2e}".format(median), 'T gates')
    print('With the specified parameters, synthesising that many T gates should take approximately', "{:0.2e}".format(c_calculator.calculate_time(median)), 'seconds')

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
# Interaction Picture: python3 main.py water interaction_picture 'C 2p' OK 
# Sublinear Scaling: python3 main.py water sublinear_scaling 'C 2p'DEPRECATED