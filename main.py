import utils
from molecule import Molecule
from pyscf.pbc import gto
import numpy as np
import scipy

#Read config file with the QFold configuration variables
config_path = './config/config.json'
tools = utils.Utils(config_path)

molecule = Molecule(name = 'water', tools = tools)

molecule.low_rank_approximation(occupied_indices = [0,1,2], active_indices = [3,4], virtual_indices = [5,6], sparsify = True)
# ne_act_cas, n_mocore, n_mocas, n_movir = molecule.active_space(ao_labels=['O 2pz'])