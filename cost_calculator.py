class Cost_calculator:

    def __init__(self, molecule, tools):

        self.molecule = molecule
        self.tools = tools

    def calculate_cost(self, method): 

        cost = 0

        if method == 'qdrift':
            print('executing qdrift ...')
        elif method == 'rand_ham':
            print('executing rand_ham ...')

        elif method == 'taylor_naive':
            print('executing taylor_naive ...')

        elif method == 'taylor_on_the_fly':
            print('executing taylor_on_the_fly ...')

        elif method == 'configuration_interaction':
            print('executing configuration_interaction ...')

        elif method == 'low_depth_trotter':
            print('executing low_depth_trotter ...')
        
        elif method == 'low_depth_taylor':
            print('executing low_depth_taylor ...')

        elif method == 'low_depth_taylor_on_the_fly':
            print('executing low_depth_taylor_on_the_fly ...')

        elif method == 'linear_t':
            print('executing linear_t ...')

        elif method == 'sparsity_low_rank':
            print('executing sparsity_low_rank ...')

        elif method == 'interaction_picture':
            print('executing interaction_picture ...')

        elif method == 'sublinear_scaling':
            print('executing sublinear_scaling ...')

        return cost