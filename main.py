import utils
from molecule import Molecule

#Read config file with the QFold configuration variables
config_path = './config/config.json'
tools = utils.Utils(config_path)

molecule = Molecule(name = 'water', tools = tools)

molecule.active_space(ao_labels=['O 2py'])