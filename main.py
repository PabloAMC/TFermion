import utils
from molecule import Molecule

#Read config file with the QFold configuration variables
config_path = './config/config.json'
tools = utils.Utils(config_path)

molecule = Molecule(name = 'water', tools = tools)

molecule.low_rank_approximation()
# ne_act_cas, n_mocore, n_mocas, n_movir = molecule.active_space(ao_labels=['O 2pz'])