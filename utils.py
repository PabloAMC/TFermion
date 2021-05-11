import argparse
import json
from molecule import CHEMICAL_ACCURACY
import numpy as np
import random as rnd
import sympy
from scipy import integrate
from scipy.optimize import NonlinearConstraint, LinearConstraint

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

        parser.add_argument("molecule_name", help="name of the molecule to analyze", type=str)
        parser.add_argument("method", help="method to calculate the energy of the molecule", type=str)
        parser.add_argument("ao_labels", help="atomic orbital labels for the avas method to select the active space. Example: ['Fe 3d', 'C 2pz']", type=list)

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

    def order_find(self, function, x0, e, xeval):
        
        x = sympy.Symbol('x')
        
        def factorial(n):

            if n <= 0:
                return 1
            else:
                return n*factorial(n-1)
        
        def taylor_err(function, x0, n, z = None):
            if z == None:
                z = x0
            a = (function.diff(x,n).subs(x,z))/(factorial(n))*(x-x0)**n
            return a

        order = 0
        te = 1
        zeta = np.linspace(x0,xeval,20)
        
        while te > e:# or order < 10:

            print("Order:", order, "error(e):", e, "calculated error(te):", te, "stop: e>te")
            order += 1
            #for z in zeta:
                #print(taylor_err(f, x0, order, z).subs(x,xeval).evalf())
            te = np.max([np.abs(taylor_err(function, x0, order, z).subs(x,xeval).evalf()) for z in zeta])
            #print('order',order, te,'\n')
            
        return order

    def f(self, x, y):
        return 1/(x**2 + y**2)
    def I(self, N0):
        return integrate.nquad(self.f, [[1, N0],[1, N0]])[0]

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

    def sum_constraint(self, x):
        return sum(x)

    def arbitrary_state_synthesis(self, N):
        return 2**(np.ceil(np.log2(N)))

    def rotation_synthesis(self, epsilon_SS):
        return (10+12*np.log2(1/epsilon_SS))

    def z_rotation(self, epsilon_SS):
        return 10 + 4*np.log2(1/epsilon_SS)

    def multi_controlled_not(self, N):
        return 16*(N-2)

    # It is necessary to generate two constraints: one linear (each value should be in the range greather than 0 and chemical_accuracy) and one non linear (errors sum should be in the range 0 and chemical accuracy)
    def generate_constraints(self, number_errors):

        
        # In the linear constraint it is necessary to define the shape of the constraint. For example, if there is three errors:
        # 0 > 1*e_1 + 0*e_2 + 0*e_3 > CHEMICAL ACCURACY     [  1,  0,  0] [e_1]
        # 0 > 0*e_1 + 1*e_2 + 0*e_3 > CHEMICAL ACCURACY     [  0,  1,  0] [e_2]
        # 0 > 0*e_1 + 0*e_2 + 1*e_3 > CHEMICAL ACCURACY     [  0,  0,  1] [e_3]
        
        shape_linear_constraint = []
        for index in range(number_errors):
            row_linear_constraint = []

            for index_row in range(number_errors):
                row_linear_constraint.append(1) if index_row == index else row_linear_constraint.append(0)

            shape_linear_constraint.append(row_linear_constraint)

        min_values_linear_constraint = [1e-10 for _ in range(number_errors)]
        max_values_linear_constraint = [CHEMICAL_ACCURACY for _ in range(number_errors)]

        linear_constraint = LinearConstraint(A=shape_linear_constraint, lb=min_values_linear_constraint, ub=max_values_linear_constraint)
        nonlinear_constraint = NonlinearConstraint(fun=self.sum_constraint, lb=0, ub=CHEMICAL_ACCURACY)

        return linear_constraint, nonlinear_constraint

    def generate_initial_error_values(self, number_errors):

        maximum_value = CHEMICAL_ACCURACY/number_errors
        minimum_value = maximum_value/2

        return [rnd.uniform(minimum_value, maximum_value) for _ in range(number_errors)]