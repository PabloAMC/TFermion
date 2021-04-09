import utils
from molecule import Molecule
from pyscf.pbc import gto
import numpy as np

#Read config file with the QFold configuration variables
config_path = './config/config.json'
tools = utils.Utils(config_path)

molecule = Molecule(name = 'NaCl', tools = tools)

molecule.low_rank_approximation(occupied_indices = [], active_indices = [], virtual_indices = [])
# ne_act_cas, n_mocore, n_mocas, n_movir = molecule.active_space(ao_labels=['O 2pz'])