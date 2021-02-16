import argparse
import json
import numpy as np
import sympy
from scipy import integrate

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

        parser = argparse.ArgumentParser(description="Tool that combines AI and QC to solve protein folding problem.\n Example: python main.py glycylglycine GG 5 minifold simulation -c")

        parser.add_argument("molecule_name", help="name of the molecule to analyze", type=str)
        parser.add_argument("method", help="method to calculate the energy of the molecule", type=str)
        
        self.args = parser.parse_args()

        if self.args.id == None: self.args.id = -1
        if self.args.cost == None: self.args.cost = -1

        return self.args

    # Taylor approximation at x0 of the function 'function'
    def taylor(self, function,x0,n):
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
        
        order = 0
        te = 1
        zeta = np.linspace(x0,xeval,20)

        while te > e:# or order < 10:
            order += 1
            #for z in zeta:
                #print(taylor_err(f, x0, order, z).subs(x,xeval).evalf())
            te = np.max([np.abs(self.taylor_err(function, x0, order, z).subs(x,xeval).evalf()) for z in zeta])
            #print('order',order, te,'\')
            
        return order

    def factorial(self, n):
        if n <= 0:
            return 1
        else:
            return n*self.factorial(n-1)
        
    def taylor_err(self, function, x0, n, z = None):
        if z == None:
            z = x0
                
        x = sympy.Symbol('x')

        #print('coefficient order',n, function.diff(x,n)/(factorial(n)))#.subs(x,z))
        a = (function.diff(x,n).subs(x,z))/(self.factorial(n))*(x-x0)**n
        #print('coefficient order',n, (function.diff(x,n).subs(x,z)/(factorial(n))*(x-x0)**n))
        #print('a',a)
        return a

    def f(self, x, y):
        return 1/(x**2 + y**2)
    def I(self, N0):
        return integrate.nquad(self.f, [[1, N0],[1, N0]])[0]

    def error_optimizer(self, eps_array):
        epsilon_PEA = eps_array[0]
        
        epsilon_S = eps_array[1]
        
        #TODO eps_tot to calculate
        epsilon_HS = eps_tot - eps_array[0] - eps_array[1]