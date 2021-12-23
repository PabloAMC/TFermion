import argparse
import json
from molecule import CHEMICAL_ACCURACY
import numpy as np
import random as rnd
import math
import sympy
from scipy import integrate
from scipy.optimize import NonlinearConstraint, LinearConstraint
from itertools import groupby
import os

BR_INITIAL_VALUE = 7

class Utils():

    def __init__(self, config_path=''):

        if config_path != '':
            try:
                f = open(config_path)
                f.close()
            except IOError:
                print('<!> Info: No configuration file')
                raise Exception('It is necessary to create a configuration file (.json) for some variables')

            with open(config_path) as json_file:
                        self.config_variables = json.load(json_file)

    def get_config_variables(self):
        return self.config_variables

    def parse_arguments(self):

        parser = argparse.ArgumentParser(description="Tool to estimate the T gates cost of many quantum energy calculator methods.\n Example: python qphase.py methane qdrift")

        parser.add_argument("molecule_info", nargs='?', default=None, help="information about molecule to analyze. It could be a name, a geometry file (with .chem extension) or a hamiltonian file (with .h5 or .hdf5 extension)", type=str)
        parser.add_argument("method", nargs='?', default=None, help="method to calculate the energy of the molecule", type=str)
        parser.add_argument("ao_labels", nargs='*', default=[], help="atomic orbital labels for the avas method to select the active space. Example: ['Fe 3d', 'C 2pz']")
        
        parser.add_argument("-c", "--charge",  help="charge of the molecule, defaults to 0", type=int)

        self.args = parser.parse_args()

        return self.args

    # Taylor approximation at x0 of the function 'function'
    def taylor(self, function, x0, n):
        i = 0
        p = 0
        x = sympy.Symbol('x')
        while i <= n:
            p = p + (function.diff(x,i).subs(x,x0))/(self.factorial(i))*(x-x0)**i
            i += 1
        return p

    #print(taylor(sympy.sqrt(x), 1, 5))#.subs(x,1).evalf())

    def order_find(self, function, e, xeval, function_name):
        
        error = 1
        # get the half digits of the xeval (len(xeval)/2)
        order = 0

        # this array stores the last n error values in order to check if all are equal (a minimum optimization point is reached)
        last_error_values = [0, 1]

        while error > e and not self.all_equal(last_error_values):

            if function_name == 'sqrt' or function_name == 'exp':
                n = int(str(xeval)[:max(int(len(str(int(xeval)))/2),1)])
                error, _ = self.calculate_error_function(function, function_name, n, xeval, order)
            elif function_name == 'cos':
                error, xeval = self.calculate_error_function(function, function_name, 1, xeval, order, xeval)

            # if maximum length is reached, last value is deleted
            if len(last_error_values) > 10: last_error_values.pop(0)
            last_error_values.append(error)

            order+=1

        return order

    def all_equal(self, iterable):
        g = groupby(iterable)
        return next(g, True) and not next(g, False)


    def calculate_error_function(self, function, function_name, n, xeval, order, value_to_find=0):
        
        if function_name == 'sqrt':
        
            n = ((xeval/n)+n)/2
            error = function(xeval)-n

            return error, xeval

        elif function_name == 'exp':

            d = xeval # d=x0-x / x0=0 and x=xeval
            error = 1

            for i in range(1, order+1):
                error *= d/i
            
            return error, xeval

        elif function_name == 'cos':

            #TODO: check if it is necessary to convert to radians
            K = 0.6072529350088812561694
            x,y = 1, 0
            d = 1.0

            if xeval < 0:
                d = -1.0

            (x,y) = (x - (d*(2.0**(-order))*y), (d*(2.0**(-order))*x) + y)
            xeval = xeval - (d*math.atan(2**(-order)))

            error = K*x - math.cos(value_to_find)

            return error, xeval

        else:
            raise NotImplementedError

    def f(self, x, y):
        return 1/(x**2 + y**2)
    def I(self, N0):
        return integrate.nquad(self.f, [[1, N0],[1, N0]])[0]
    def sum_1_over_nu(self,N):
        return 2*np.pi*N**(2/3) # based on integrate(r*sin(t), (r, 0, N), (p, 0, 2*pi), (t, 0, pi)) and eq 13 in https://www.nature.com/articles/s41534-019-0199-y
    def sum_1_over_nu_squared(self,N):
        return 4*np.pi*N**(1/3) # based on integrate(sin(t), (r, 0, N), (p, 0, 2*pi), (t, 0, pi)) and eq 13 in https://www.nature.com/articles/s41534-019-0199-y


    def bisection(self, symbol, expr, upper_bound = 1e10, lower_bound = 100):
        top = upper_bound
        bottom = lower_bound
        while top-bottom > 1:
            eval_r = 2 ** (np.log2(top)+np.log2(bottom)/2)
            result_r = expr.evalf(subs={symbol: eval_r})
            if result_r < eval_r:
                top = eval_r
            else:
                bottom = eval_r
        return eval_r

    def fun_constraint(self, x):

        if self.config_variables['error_optimization_function'] == 'sum':
            return sum(x[:self.number_errors])

        elif self.config_variables['error_optimization_function'] == 'rmse':
            return math.sqrt(sum([x_value**2 for x_value in x[:self.number_errors]]))

        else:
            raise Exception("Function to optimize constraints not recognized")

    def arbitrary_state_synthesis(self, n):
        '''
        Number of rotations in arbitrary state synthesis
        Use theorems 8 and 9 from https://ieeexplore.ieee.org/abstract/document/1629135
        n is the size of the register
        '''
        return 2*2**(n)-2

    def pauli_rotation_synthesis(self, epsilon_SS):
        result = 10 + 4*np.ceil(np.log2(1/epsilon_SS))
        return result
    def c_pauli_rotation_synthesis(self, epsilon_SS):
        return 2*self.pauli_rotation_synthesis(epsilon_SS)

    def SU2_rotation_synthesis(self, epsilon_SS):
        return 3*self.pauli_rotation_synthesis(epsilon_SS)

    def c_SU2_rotation_synthesis(self, epsilon_SS):
        return 2*self.SU2_rotation_synthesis(epsilon_SS)

    def multi_controlled_not(self, N):
        return 16*(N-2)

    def sum_cost(self, n):
        return 4*n

    def multiplication_cost(self, n):
        return 21*n**2

    def divide_cost(self, n):
        return 14*n**2+7*n

    def compare_cost(self, n):
        return 8*n
    

    def generate_optimization_conditions(self, parameters_to_optimize, chemical_acc_modifier):

        constraints = []
        initial_values = []

        constraints = [self.generate_linear_constraints(parameters_to_optimize, chemical_acc_modifier)] + [self.generate_non_linear_constraint(chemical_acc_modifier)]

        initial_values += self.generate_initial_error_values(parameters_to_optimize, chemical_acc_modifier)

        return constraints, initial_values

    def generate_linear_constraints(self, parameters_to_optimize, chemical_acc_modifier):

        self.number_errors = 0

        min_values_linear_constraint = []
        max_values_linear_constraint = []
        
        # In the shape constraint it is necessary to define the shape of the constraint. For example, if there is three errors:
        # 0 > 1*e_1 + 0*e_2 + 0*e_3 > CHEMICAL ACCURACY     [  1,  0,  0] [e_1]
        # 0 > 0*e_1 + 1*e_2 + 0*e_3 > CHEMICAL ACCURACY     [  0,  1,  0] [e_2]
        # 0 > 0*e_1 + 0*e_2 + 1*e_3 > CHEMICAL ACCURACY     [  0,  0,  1] [e_3]
        shape_constraint = np.diag(np.full(len(parameters_to_optimize),1))

        for param in parameters_to_optimize:

            if 'epsilon' in param:
                min_values_linear_constraint.append(1e-10)
                max_values_linear_constraint.append(CHEMICAL_ACCURACY*chemical_acc_modifier)
                self.number_errors += 1

            elif 'br' == param:
                min_values_linear_constraint.append(BR_INITIAL_VALUE/2)
                max_values_linear_constraint.append(BR_INITIAL_VALUE*2)

        return LinearConstraint(A=shape_constraint, lb=min_values_linear_constraint, ub=max_values_linear_constraint)

    # It is necessary to generate two constraints: one linear (each value should be in the range greather than 0 and chemical_accuracy) and one non linear (errors sum should be in the range 0 and chemical accuracy)
    def generate_non_linear_constraint(self, chemical_acc_modifier):

        nonlinear_constraint = NonlinearConstraint(fun=self.fun_constraint, lb=0, ub=CHEMICAL_ACCURACY*chemical_acc_modifier)

        return nonlinear_constraint

    def generate_br_constraint(self):

        lower_bound = BR_INITIAL_VALUE/2
        upper_bound = BR_INITIAL_VALUE*2

        return LinearConstraint(A=[1], lb=lower_bound, ub=upper_bound)

    def generate_initial_error_values(self, parameters_to_optimize, chemical_acc_modifier):

        initial_values = []
        number_errors = len([s for s in parameters_to_optimize if "epsilon" in s])

        for param in parameters_to_optimize:
            initial_values += [rnd.uniform((CHEMICAL_ACCURACY*chemical_acc_modifier/number_errors)/2, CHEMICAL_ACCURACY*chemical_acc_modifier/number_errors)] if 'epsilon' in param else [rnd.uniform(BR_INITIAL_VALUE/2, BR_INITIAL_VALUE*2)]

        return initial_values

    def parse_geometry_file(self, molecule_info):

        with open(molecule_info) as json_file: return json.load(json_file['atoms'])

    def check_molecule_info(self, molecule_info):

        if molecule_info == "":
            return None

        # the hamiltonian is given by a path containing files eri_li.h5 and eri_li_cholesky.h5
        if os.path.isdir(molecule_info):

            if os.path.isfile(molecule_info + 'eri_li.h5') and os.path.isfile(molecule_info + 'eri_li_cholesky.h5'):
                return "hamiltonian"
            else:
                print("<*> ERROR: The given path does not contain the files eri_li.h5 and eri_li_cholesky.h5 needed for hamiltonian input")
                return "error"

        else:

            index_last_dot = molecule_info[::-1].find('.')

            # there is no dot, so no extension. Therefore, it is a name
            if index_last_dot == -1:
                return 'name'

            # there is a dot, so it is a file with extension
            else:

                # get the extension of the file taking the character from last dot
                extension = molecule_info[-index_last_dot:]

                if extension == 'geo':
                    return 'geometry'

                else:
                    print('<*> ERROR: extension in molecule information not recognized. It should be .chem (geometry) or .h5/.hdf5 (hamiltonian). The molecule name can not contain dots')