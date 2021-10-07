import openfermion
import copy
import json
import ast
import importlib
import sys
from openfermion.transforms.opconversions.term_reordering import normal_ordered

from openfermionpsi4 import run_psi4
from openfermionpyscf import run_pyscf
from openfermionpyscf._run_pyscf import prepare_pyscf_molecule, compute_integrals, compute_scf

from openfermion.utils import Grid
import openfermion.ops.representations as reps
from openfermion.chem import geometry_from_pubchem, MolecularData
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.hamiltonians import plane_wave_hamiltonian, jordan_wigner_dual_basis_hamiltonian 
from openfermion.hamiltonians import dual_basis_external_potential, plane_wave_external_potential
from openfermion.hamiltonians import dual_basis_potential, plane_wave_potential
from openfermion.hamiltonians import dual_basis_kinetic, plane_wave_kinetic
from openfermion.transforms  import  get_fermion_operator, get_majorana_operator, get_interaction_operator, normal_ordered, get_molecular_data, jordan_wigner
from openfermion.circuits import low_rank_two_body_decomposition
from openfermion.ops.operators import fermion_operator


from pyscf import gto, scf, mcscf, fci, ao2mo
from pyscf.mcscf import avas
from pyscf.lib.parameters import BOHR


import numpy as np
import copy
import time
import scipy 

'''
The active space selection method avas is not supported with periodic boundary conditions.
Similarly, the low rank approximation of the Hamiltonian is not possible in that case since 
low_rank_two_body_decomposition requires the two body integrals of the Hamiltonian, which are not
available for Cell objects from Pyscf. This means that periodic materials cannot be modelled accurately.
Nevertheless, some important links for the periodic boundary condition (not used) are:

1. Crystal cell structure: https://sunqm.github.io/pyscf/modules/pbc/gto.html#pbc-gto
                see also:  https://github.com/pyscf/pyscf/blob/1de8c145abb3e1a7392df9118e8062e6fe6bde00/pyscf/pbc/gto/cell.py#L1010

2. Costumizing the kpts Hamiltonian (for periodic boundary conditions): https://github.com/pyscf/pyscf-doc/blob/master/examples/pbc/40-customize_kpts_hamiltonian.py
3. Hartree Fock algorithm for that Hamiltonian: https://github.com/pyscf/pyscf/blob/master/pyscf/pbc/scf/khf.py
'''

CHEMICAL_ACCURACY = 0.0015 #in Hartrees, according to http://greif.geo.berkeley.edu/~driver/conversions.html
tol = 1e-8
#wigner_seitz_radius = 5. # Chosen as in https://quantumai.google/openfermion/tutorials/circuits_2_diagonal_coulomb_trotter, but may not make sense

class Molecule:

    def __init__(self, molecule_info, molecule_info_type, tools, charge = 0, program = 'pyscf'):

        self.molecule_info = molecule_info
        self.tools = tools
        self.program = program

        # molecule info could be a name, geometry information or hamiltonian description
        self.molecule_info_type = molecule_info_type

        if self.molecule_info_type == 'name':
            self.molecule_info = self.molecule_info.replace('_', ' ')
            molecule_geometry = geometry_from_pubchem(self.molecule_info)

        elif self.molecule_info_type == 'geometry':
            molecule_geometry = None
            with open(molecule_info) as json_file: 
                molecule_geometry = json.load(json_file)['atoms']

        [self.molecule_geometry, self.molecule_data] = self.calculate_geometry_params(molecule_geometry, charge)

        #Add possibility of boundary conditions https://sunqm.github.io/pyscf/tutorial.html#initializing-a-crystal -> Seems quite complicated and not straightforward
        if program == 'psi4': 
            self.molecule_psi4 = run_psi4(self.molecule_data,run_scf=True, run_mp2=True, run_fci=False)

        elif program == 'pyscf':
            self.molecule_pyscf = run_pyscf(self.molecule_data,run_scf=True, run_mp2=True, run_ccsd=True)
            print('<i> HF energy, MP2 energy, CCSD energy', self.molecule_pyscf.hf_energy, self.molecule_pyscf.mp2_energy, self.molecule_pyscf.ccsd_energy)

        self.occupied_indices = None
        self.active_indices = None #range(self.molecule_data.n_orbitals) # This is the default
        self.virtual_indices = []

        self.N  = self.molecule_data.n_orbitals * 2 # The 2 is due to orbitals -> spin orbitals

        #self.build_grid()
        #self.get_basic_parameters()

    def calculate_geometry_params(self, molecule_geometry, charge):

        ## Center the molecule so that coords can be put in a box

        # Tuple to list
        for i, (at, coord) in zip(range(len(molecule_geometry)), molecule_geometry):
            molecule_geometry[i] = (at, list(coord))

        self.xmax = 0

        # Shift each coordinate
        for j in range(3):
            maximum = max([molecule_geometry[i][1][j] for i in range(len(molecule_geometry))])
            minimum = min([molecule_geometry[i][1][j] for i in range(len(molecule_geometry))])

            avg = (maximum + minimum)/2

            for i in range(len(molecule_geometry)):
                molecule_geometry[i][1][j] -= avg

            maximum = max([molecule_geometry[i][1][j] for i in range(len(molecule_geometry))])
            self.xmax = max(self.xmax, maximum) 
        
        # List to tuple
        for i, (at, coord) in zip(range(len(molecule_geometry)), molecule_geometry):
            molecule_geometry[i] = (at, tuple(coord))

        #From OpenFermion
        return [molecule_geometry, MolecularData(molecule_geometry, self.tools.config_variables['basis'], charge = charge, multiplicity = 1, filename = 'name')]
    

    def get_basic_parameters(self):

        self.eta = self.molecule_data.n_electrons
        if self.occupied_indices and self.active_indices:
            _, one_body_integrals, two_body_integrals = self.molecule_data.get_active_space_integrals(occupied_indices=self.occupied_indices, 
                                                                                                active_indices=self.active_indices)
        else:
            one_body_integrals, two_body_integrals = self.molecule_data.get_integrals()
        self.lambda_value, self.Lambda_value, self.Gamma = self.get_one_norm_int_woconst(one_body_integrals,
                                                                                        two_body_integrals)


    def build_grid(self, grid_length: int = 7):
        '''
        non_periodic: If False, impose periodic boundary conditions
        '''

        self.N_grid = grid_length**3
        self.eta = self.molecule_data.n_electrons
        
        length_scale = 4*self.xmax # We set a box whose length is 4 times the maximum coordinate value of any atom, as the box has to be twice as large as the maximum distance in each coord
        grid = Grid(dimensions = 3, length = grid_length, scale = length_scale) # Complexity is determined by lenght

        JW_op = jordan_wigner_dual_basis_hamiltonian(grid, self.molecule_geometry, spinless = True)

        l = abs(np.array(list(JW_op.terms.values())))

        self.lambda_value_grid = sum(l[1:])
        self.Lambda_value_grid = max(l[1:])
        self.Gamma_grid = np.count_nonzero(l[:1]>tol)
        
        self.Omega = grid.volume

        return grid

        # recursive method that iterates over all rows of a molecule to get the parameters:
        # lambda_value is the sum all coefficients of the hamiltonian (sum of all terms)
        # Lambda_value is the maximum value of all terms
        # N is the number of orbitals
        # gamma is the total number of elements (without counting values under some threshold)

    def active_space(self, ao_labels):
        '''
        Inputs:
        ao_labels: list #atomic orbitals needed to construct the active space. EXAMPLE: ao_labels = ['Fe 3d', 'C 2pz'] https://github.com/pyscf/pyscf/blob/18030c75a5c69c1da84574d111693074a622de56/pyscf/gto/mole.py#L1511

        Avas example taken from https://github.com/pyscf/pyscf-doc/blob/93f34be682adf516a692e28787c19f10cbb4b969/examples/mcscf/43-avas.py
        Avas documentation reference: https://github.com/pyscf/pyscf/blob/5796d1727808c4ab6444c9af1f8af1fad1bed450/pyscf/mcscf/avas.py
        Inspired by the function run_pyscf from OpenFermion-Pyscf https://github.com/quantumlib/OpenFermion-PySCF/blob/60ddc080226e89ea5a30c4a5238b1e5418e00440/openfermionpyscf/_run_pyscf.py#L100
        Restricting the molecule to the active space: https://quantumai.google/reference/python/openfermion/chem/MolecularData#get_active_space_integrals

        Objects we use
        molecule_data: MolecularData https://quantumai.google/reference/python/openfermion/chem/MolecularData
        molecule_pyscf: PyscfMolecularData https://github.com/quantumlib/OpenFermion-PySCF/blob/8b8de945db41db2b39d588ff0396a93566855247/openfermionpyscf/_pyscf_molecular_data.py#L23
        pyscf_mol : A pyscf molecule instance https://github.com/pyscf/pyscf/blob/master/pyscf/gto/mole.py
        pyscf_scf: scf method https://github.com/pyscf/pyscf/blob/7be5e015b2b40181755c71d888449db936604660/pyscf/scf/__init__.py#L123
        pyscf_mcscf: mcscf method https://github.com/pyscf/pyscf/blob/7be5e015b2b40181755c71d888449db936604660/pyscf/mcscf/__init__.py#L193
        
        Returns:
        - occupied_indices
        - active_indices
        These indices can be used in self.get_basic_parameters(). 
        Also modifies self.molecule_data and self.molecule_pyscf in place.
        '''
        ao_labels = ast.literal_eval(ao_labels)

        # Selecting the active space
        pyscf_scf = self.molecule_pyscf._pyscf_data['scf'] #similar to https://github.com/quantumlib/OpenFermion-PySCF/blob/8b8de945db41db2b39d588ff0396a93566855247/openfermionpyscf/_pyscf_molecular_data.py#L47
        my_avas = avas.AVAS(pyscf_scf, ao_labels, canonicalize=False)
        n_mocas, ne_act_cas, mo_coeff = my_avas.kernel()

        n_mocore = my_avas.occ_weights.shape[0] - n_mocas
        n_movir = my_avas.vir_weights.shape[0]

        pyscf_scf.mo_coeff = mo_coeff
        # mo_occ = pyscf_scf.mo_occ contains some information on the occupation

        # Correcting molecular coefficients 
        self.molecule_data.canonical_orbitals = mo_coeff.astype(float)
        self.molecule_pyscf._canonical_orbitals = mo_coeff.astype(float)
        self.molecule_data._pyscf_data['scf'] = pyscf_scf

        # Get two electron integrals
        pyscf_mol = self.molecule_data._pyscf_data['mol']
        one_body_integrals, two_body_integrals = compute_integrals(pyscf_mol, pyscf_scf)
        self.molecule_data.one_body_integrals = one_body_integrals
        self.molecule_data.two_body_integrals = two_body_integrals

        self.molecule_data.overlap_integrals = pyscf_scf.get_ovlp()

        # This does not give the natural orbitals. If those are wanted check https://github.com/pyscf/pyscf/blob/7be5e015b2b40181755c71d888449db936604660/pyscf/mcscf/__init__.py#L172
        # Complete Active Space Self Consistent Field (CASSCF), an option of Multi-Configuration Self Consistent Field (MCSCF) calculation. A more expensive alternative would be Complete Active Space Configuration Interaction (CASCI)
        pyscf_mcscf = mcscf.CASSCF(pyscf_scf, n_mocas, ne_act_cas).run(mo_coeff) #Inspired by the mini-example in avas documentation link above

        self.molecule_data._pyscf_data['mcscf'] = pyscf_mcscf
        self.molecule_data.mcscf_energy = pyscf_mcscf.e_tot

        self.molecule_data.orbital_energies = pyscf_mcscf.mo_energy.astype(float)
        self.molecule_data.canonical_orbitals = pyscf_mcscf.mo_coeff.astype(float)

        self.occupied_indices = list(range(n_mocore))
        self.active_indices = list(range(n_mocore, n_mocas))
        self.virtual_indices = list(range(n_mocas, n_movir))

        return ne_act_cas

    def sparsify(self, occupied_indices, virtual_indices):
        '''Unused, see low_rank approximation'''

        pyscf_scf = self.molecule_data._pyscf_data['scf']
        pyscf_mol = self.molecule_data._pyscf_data['mol']

        two_body_integrals = self.molecule_data.two_body_integrals
        one_body_integrals = self.molecule_data.one_body_integrals

        def sparsification_mp2_energy(threshold):

            mol = gto.M()
            mol.nelectron = self.molecule_pyscf.n_electrons

            mf = scf.RHF(mol)

            h_core = pyscf_scf.get_hcore()
            h_core[h_core < threshold] = 0.
            mf.get_hcore = lambda *args: h_core
            mf.get_ovlp = lambda *args: pyscf_scf.get_ovlp()
            # ao2mo.restore(8, eri, n) to get 8-fold permutation symmetry of the integrals
            # ._eri only supports the two-electron integrals in 4-fold or 8-fold symmetry.
            # See http://vergil.chemistry.gatech.edu/notes/permsymm/permsymm.pdf

            eri = pyscf_scf._eri
            eri[eri < threshold] = 0.
            mf._eri = eri if abs(eri) > threshold else 0

            mf.kernel()

            mol.incore_anyway = True

            # If there is an active space we want to work with in the Moller Plesset energy calculation, we can do it here
            if occupied_indices and virtual_indices:
                pt = mf.MP2().set(frozen = occupied_indices + virtual_indices).run()
            else:
                pt = mf.MP2().set().run()

            energy = pt.e_tot
            return energy

        # Until here------------------------------------ Iterate to see how high can we put the threshold without damaging the energy estimates (error up to chemical precision)
        exact_E = sparsification_mp2_energy(threshold = 0)

        nconstraint = scipy.optimize.NonlinearConstraint(fun = lambda threshold: sparsification_mp2_energy(threshold) - exact_E, lb = -CHEMICAL_ACCURACY, ub = +CHEMICAL_ACCURACY)
        lconstraint = scipy.optimize.LinearConstraint(A = np.array([1]), lb = 1e-10, ub = 1)
        result = scipy.optimize.minimize(fun = lambda threshold: 1e-2/(threshold+1e-4), x0 = 1e-4, constraints = [nconstraint, lconstraint], tol = .01*CHEMICAL_ACCURACY, options = {'maxiter': 50}, method='COBYLA') # Works with COBYLA, but not with SLSQP (misses the boundaries) or trust-constr (oscillates)
        threshold = float(result['x'])
        approximate_E = sparsification_mp2_energy(threshold = threshold)

        two_body_integrals[abs(two_body_integrals) < threshold] = 0.
        one_body_integrals[abs(one_body_integrals) < threshold] = 0.
        one_body_coefficients, two_body_coefficients = spinorb_from_spatial(one_body_integrals, two_body_integrals)
        constant = self.molecule_data.nuclear_repulsion

        pTensor = reps.PolynomialTensor({(): constant, (1, 0): one_body_coefficients, (1, 1, 0, 0): two_body_coefficients})
        Maj_op = get_majorana_operator(pTensor)
        l_maj = np.abs(np.array(list(Maj_op.terms.values())))
        lambda_value_low_rank = sum(l_maj[1:])

        return lambda_value_low_rank, threshold

    def low_rank_approximation(self, sparsify = False):
        '''
        Aim: get a low rank (rank-truncated) hamiltonian such that the error using say mp2 is smaller than chemical accuracy. Then use that Hamiltonian to compute the usual terms
        
        Args: 
        occupied_indices: list = []
        active_indices: list = []
        virtual_indices: list = []
        sparsify: bool

        Returns:
        molecular_hamiltonian: MolecularOperator # Truncated Hamiltonian
        final_rank: int # Rank of the truncated Hamiltonian

        Basic strategy:
            - Perform Low-Rank trucation
            - Use Low-Rank truncated Hamiltonian to create pyscf_mol object (named mol)
            - Compute Moller-Plesset (total) ground state energy of the pyscf_mol object, in the active space if provided
            - Iterate the previous process using some numeric method such that the low-rank trucation does not significantly affect the energy computed by MP2 (Chemical accuracy)
            - Use the threshold computed previously to perform the low-rank approximation in the CAS Hamiltonian (Hamiltonian restricted to active orbitals)
            - Prepare OpenFermion's Molecular Hamiltonian Operator from the CAS Hamiltonian

        If sparsify: # WARNING: optimization will be significantly slower
            - The thresholds for the low rank approximation and the sparsity are optimized as a function of lambda parameter

        Perform low rank approximation using
        https://github.com/quantumlib/OpenFermion/blob/4781602e094699f0fe0844bcded8ef0d45653e81/src/openfermion/circuits/low_rank.py#L76
        How precise it is using MP2: 
        https://github.com/pyscf/pyscf/blob/c9aa2be600d75a97410c3203abf35046af8ca615/pyscf/mp/mp2.py#L411 (also https://github.com/psi4/psi4numpy)
        Costumizing Hamiltonian: https://github.com/pyscf/pyscf-doc/blob/master/examples/scf/40-customizing_hamiltonian.py    
        See also a discussion on this topic: https://github.com/quantumlib/OpenFermion/issues/708
        
        To get active space Hamiltonian in OpenFermion use https://quantumai.google/reference/python/openfermion/chem/MolecularData#get_active_space_integrals
        To restrict Moller-Plesset 2nd order calculation to the Chosen Active Space, https://github.com/pyscf/pyscf/blob/5796d1727808c4ab6444c9af1f8af1fad1bed450/pyscf/mp/mp2.py#L411
            see also https://github.com/pyscf/pyscf/blob/5796d1727808c4ab6444c9af1f8af1fad1bed450/pyscf/mp/__init__.py#L25
        To create a molecular_hamiltonian (MolecularOperator class) https://github.com/quantumlib/OpenFermion/blob/40f4dd293d3ac7759e39b0d4c061b391e9663246/src/openfermion/chem/molecular_data.py#L878

        To perform optimization (use COBYLA (default) or trust-constr): https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

        Probably beyond our interest: (if we wanted to create a pyscf_mol object with the truncated Hamiltonian, which we have skipped over)
        One can truncate the basis using something similar to https://github.com/pyscf/pyscf/blob/9c8b06d481623b50ccdca9f88d833de7320ac3cd/examples/gto/04-input_basis.py#L98
        To select get the ao_labels: https://github.com/pyscf/pyscf/blob/5796d1727808c4ab6444c9af1f8af1fad1bed450/pyscf/gto/mole.py#L1526
        To get the overlap matrix https://github.com/pyscf/pyscf/blob/5796d1727808c4ab6444c9af1f8af1fad1bed450/pyscf/mcscf/avas.py#L128
        '''

        occupied_indices = self.occupied_indices
        active_indices = self.active_indices 
        virtual_indices = self.virtual_indices

        pyscf_scf = self.molecule_data._pyscf_data['scf']
        pyscf_mol = self.molecule_data._pyscf_data['mol']

        if occupied_indices or virtual_indices:
            new_core_constant, new_one_body_integrals, new_two_body_integrals = self.molecule_data.get_active_space_integrals(occupied_indices = occupied_indices, active_indices = active_indices)
        else:
            new_core_constant = 0
            new_two_body_integrals = self.molecule_data.two_body_integrals          # electronic repulsion integrals
            new_one_body_integrals = self.molecule_data.one_body_integrals

        # From here------------------------------------------------

        def low_rank_truncation_mp2_energy(rank_threshold, sparsity_threshold):

            one_b_tensor, two_b_tensor = spinorb_from_spatial(new_one_body_integrals, new_two_body_integrals)
            two_b_tensor[np.abs(two_b_tensor) < sparsity_threshold] = 0.
        
            lambda_ls, one_body_squares, one_body_correction, truncation_value = low_rank_two_body_decomposition(two_b_tensor,
                                                                                                    truncation_threshold=rank_threshold,
                                                                                                    final_rank=None,
                                                                                                    spin_basis=True)

            # Electronic Repulsion Integral
            two_b_tensor = np.einsum('l,lpq,lrs->pqrs',lambda_ls, (one_body_squares + np.transpose(one_body_squares, (0,2,1)))/2, (one_body_squares + np.transpose(one_body_squares, (0,2,1)))/2)

            # Tensors have type complex but they do not have imaginary part
            two_body_tensor = np.real_if_close(two_b_tensor)
            assert(np.isreal(two_body_tensor).all())

            one_body_tensor = one_b_tensor + one_body_correction
            one_body_tensor[abs(one_body_tensor) < sparsity_threshold] = 0.

            # Taking anticommutation relations into account
            normal_two_body_tensor = -np.transpose(two_body_tensor, (0,2,1,3))
            normal_one_body_tensor = one_body_tensor + np.einsum('pqqr-> pr', two_body_tensor)

            h_core, eri = self.spatial_from_spinorb(normal_one_body_tensor, normal_two_body_tensor)

            # Checking some of the symmetries: http://vergil.chemistry.gatech.edu/notes/permsymm/permsymm.pdf
            assert(np.isclose(eri, np.transpose(eri, (3,2,1,0)), rtol = 1e-3).all() and np.isclose(eri, np.transpose(eri, (2,3,0,1)), rtol = 1e-3).all() and np.isclose(eri, np.transpose(eri, (1,0,3,2)), rtol = 1e-3).all())

            mol = gto.M()
            mol.nelectron = self.molecule_data.n_electrons
            mf = scf.RHF(mol)

            # pyscf_scf.get_hcore() should be the same as one_body_integrals
            mf.get_hcore = lambda *args: h_core
            if active_indices:
                mf.get_ovlp = lambda *args: self.molecule_data.overlap_integrals[np.ix_(active_indices, active_indices)]
            else:
                mf.get_ovlp = lambda *args: self.molecule_data.overlap_integrals #pyscf_scf.get_ovlp() # todo: is the overlap matrix from pyscf the correct one?
            #mf.get_hcore
            # ao2mo.restore(8, eri, n) to get 8-fold permutation symmetry of the integrals
            # ._eri only supports the two-electron integrals in 4-fold or 8-fold symmetry.

            mf._eri = eri # ao2mo.restore(8, eri, mol.nelectron)

            mf.kernel()
            mol.incore_anyway = True

            # If there is an active space we want to work with in the Moller Plesset energy calculation, we can do it here
            if occupied_indices and virtual_indices:
                pt = mf.MP2().set(frozen = occupied_indices + virtual_indices).run()
            else:
                pt = mf.MP2().set().run()

            energy = pt.e_tot
            return energy

        # Until here------------------------------------ Iterate to see how high can we put the threshold without damaging the energy estimates (error up to chemical precision)
        
        def compute_lambda(threshold):
            '''Function that computes lambda as a function of the sparsity and low rank threshold'''

            rank_threshold = threshold[0]
            sparsity_threshold = threshold[1]

            one_body_coefficients, two_b_tensor = spinorb_from_spatial(new_one_body_integrals, new_two_body_integrals)
            two_b_tensor[abs(two_b_tensor) < sparsity_threshold] = 0.

            lambda_ls, one_body_squares, one_body_correction, _ = low_rank_two_body_decomposition(two_b_tensor,
                                                                                                truncation_threshold=rank_threshold,
                                                                                                final_rank=None,
                                                                                                spin_basis=True)


            one_body_coefficients = one_body_correction + one_body_coefficients
            one_body_coefficients[abs(one_body_coefficients) < sparsity_threshold] = 0.

            # Eq 10 in https://quantum-journal.org/papers/q-2019-12-02-208/
            lambda_T = np.sum(abs(one_body_coefficients))
            lambda_W = np.sum([abs(lambda_ls[i])*(np.sum(abs(one_body_squares[i])))**2 for i in range(len(lambda_ls))])
            lambda_value_low_rank = lambda_T + lambda_W

            return lambda_value_low_rank
        
        exact_E = low_rank_truncation_mp2_energy(rank_threshold = 0, sparsity_threshold = 0)

        if sparsify: 
            nconstraint = scipy.optimize.NonlinearConstraint(fun = lambda threshold: low_rank_truncation_mp2_energy(threshold[0], threshold[1]) - exact_E, lb = -CHEMICAL_ACCURACY, ub = +CHEMICAL_ACCURACY)
            lconstraint = scipy.optimize.LinearConstraint(A = np.array([[1,0],[0,1]]), lb = [1e-10,1e-10], ub = [1,1])
            result = scipy.optimize.minimize(fun = compute_lambda, x0 = [1e-8, 1e-10], constraints = [nconstraint, lconstraint], options = {'maxiter': 50, 'catol': .01*CHEMICAL_ACCURACY}, tol = 0.1, method='COBYLA') # Works with COBYLA, but not with SLSQP (misses the boundaries) or trust-constr (oscillates)
            rank_threshold = float(result['x'][0])
            sparsity_threshold = float(result['x'][1])
        else:
            nconstraint = scipy.optimize.NonlinearConstraint(fun = lambda rank_threshold: low_rank_truncation_mp2_energy(rank_threshold, 0) - exact_E, lb = -CHEMICAL_ACCURACY, ub = +CHEMICAL_ACCURACY)
            lconstraint = scipy.optimize.LinearConstraint(A = np.array([1]), lb = 1e-10, ub = 1)
            result = scipy.optimize.minimize(fun = lambda rank_threshold: 1e-2/(rank_threshold+1e-4), x0 = 1e-4, constraints = [nconstraint, lconstraint], tol = 0.1, options = {'maxiter': 50, 'catol': .01*CHEMICAL_ACCURACY}, method='COBYLA') # Works with COBYLA, but not with SLSQP (misses the boundaries) or trust-constr (oscillates)
            rank_threshold = float(result['x'])
            sparsity_threshold = 0.
            
        self.lambda_value_low_rank = compute_lambda([rank_threshold, sparsity_threshold])
        #approximate_E = low_rank_truncation_mp2_energy(rank_threshold = rank_threshold, sparsity_threshold = sparsity_threshold)

        one_body_coefficients, two_b_tensor = spinorb_from_spatial(new_one_body_integrals, new_two_body_integrals)

        if sparsify:
            two_b_tensor[abs(two_b_tensor) < sparsity_threshold] = 0.

        lambda_ls, one_body_squares, one_body_correction, _ = low_rank_two_body_decomposition(two_b_tensor,
                                                                                            truncation_threshold=rank_threshold,
                                                                                            final_rank=None,
                                                                                            spin_basis=True)

        final_rank = len(lambda_ls)

        one_body_coefficients = one_body_correction + one_body_coefficients
        if sparsify:
            one_body_coefficients[abs(one_body_coefficients) < sparsity_threshold] = 0.

        # Eq 10 in https://quantum-journal.org/papers/q-2019-12-02-208/
        lambda_T = np.sum(abs(one_body_coefficients))
        lambda_W = np.sum([abs(lambda_ls[i])*(np.sum(abs(one_body_squares[i])))**2 for i in range(final_rank)])
        self.lambda_value_low_rank = lambda_T + lambda_W 

        # The original formula is (2L+1)*(N^4/8+ N/4). Here we have to count only the non-zero elements
        self.sparsity_d = np.count_nonzero(one_body_coefficients-np.diag(np.diag(one_body_coefficients)))/2 + np.count_nonzero(np.diag(np.diag(one_body_coefficients)))
        for i in range(len(lambda_ls)):
            self.sparsity_d += 2*np.all(lambda_ls[i])*( np.count_nonzero(one_body_squares[i,:,:]-np.diag(np.diag(one_body_squares[i,:,:])))/2 + np.count_nonzero(np.diag(np.diag(one_body_squares[i,:,:]))))

        self.final_rank = final_rank

    def molecular_orbital_parameters(self):
        '''
        Returns:
        phi_max: max value reached by molecular orbitals (mo) (Used in Taylor naive and Configuration Interaction paper)
        dphi_max: maximum "directional" derivative of molecular orbitals (Used in Taylor naive and Configuration Interaction paper)
        grad_max: maximum norm of gradient (used in Configuration Interaction article)
        hess_max: maximum norm of the hessian (used in Configuration Intearction article)
        lapl_max: absolute value of the laplacian (used in Configuration Intearction article)

        - COMPUTE MO VALUES AND THEIR DERIVATIVES IN THE SPACE: https://github.com/pyscf/pyscf/blob/master/examples/gto/24-ao_value_on_grid.py (It's totally awesome that this exists)
        
        Procedure
        We evaluate the molecular orbital functions and their derivatives in random points and return the highest value

        Other relevant bibliography
        - To calculate the ao basis. Bibliography https://onlinelibrary.wiley.com/doi/pdf/10.1002/wcms.1123?casa_token=M0hDMDgf0VkAAAAA:qOQVt0GDe2TD7WzAsoHCq0kLzNgAQFjssF57dydp1rsr4ExjZ1MEP75eD4tkjpATrpkd81qnWjJmrA
        - For conversion of ao to mo
        https://github.com/pyscf/pyscf/tree/5796d1727808c4ab6444c9af1f8af1fad1bed450/pyscf/ao2mo
        https://github.com/pyscf/pyscf-doc/tree/93f34be682adf516a692e28787c19f10cbb4b969/examples/ao2mo
        '''

        pyscf_scf = self.molecule_data._pyscf_data['scf']
        pyscf_mol = self.molecule_data._pyscf_data['mol']

        coord = np.empty((0, 3))
        for _, at_coord in self.molecule_data.geometry:
            coord = np.vstack((coord, np.array(at_coord) + 10 * BOHR * np.random.random((1000,3)))) # Random coords around the atomic positions

        # deriv=2: value + gradients + second order derivatives
        ao_p = pyscf_mol.eval_gto('GTOval_sph_deriv2', coord) # (10,Ngrids,n_mo) array

        ao = ao_p[0]
        ao_grad = ao_p[1:4]  # x, y, z
        ao_hess = ao_p[4:10] # xx, xy, xz, yy, yz, zz

        mo = ao.dot(pyscf_scf.mo_coeff)
        mo_grad = np.apply_along_axis(func1d =  lambda x: x.dot(pyscf_scf.mo_coeff), axis = 2, arr = ao_grad)
        mo_hess = np.apply_along_axis(func1d =  lambda x: x.dot(pyscf_scf.mo_coeff), axis = 2, arr = ao_hess)

        def hessian_vector_norm(vec):
            assert(len(vec) == 6)
            A = np.zeros((3,3))

            A[0,0] = vec[0]
            A[0,1], A[1,0] = vec[1], vec[1]
            A[0,2], A[2,0] = vec[2], vec[2]
            A[1,1] = vec[3]
            A[1,2], A[2,1] = vec[4], vec[4]
            A[2,2] = vec[5]

            return np.linalg.norm(A, ord = 2)

        def laplacian_vector_abs(vec):
            return abs(sum([vec[0], vec[3], vec[5]]))

        self.phi_max = np.max(np.abs(mo))
        self.dphi_max = np.max(np.abs(mo_grad)) # Different from grad_max because it's the absolute value of the maximum entry (as opposed to sum of entries) of the gradient

        mo_grads_norms = np.apply_along_axis(func1d = np.linalg.norm, axis = 0, arr = mo_grad)
        mo_hess_norms = np.apply_along_axis(func1d = hessian_vector_norm, axis = 0, arr = mo_hess)
        mo_laplacian_norms = np.apply_along_axis(func1d = laplacian_vector_abs, axis = 0, arr = mo_hess)

        self.grad_max = np.max(mo_grads_norms)
        self.hess_max = np.max(mo_hess_norms)
        self.lapl_max = np.max(mo_laplacian_norms)

        return

    def calculate_zeta_max_i(self):
        '''Returns the charge of the larger atom in the molecule'''
        zeta_max_i = 0

        # The Periodic Table as a python list and dictionary.
        periodic_table = [  #
            '?', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al',
            'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe',
            'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y',
            'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
            'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
            'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir',
            'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac',
            'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No',
            'Lr'
        ]

        for item in self.molecule_geometry:
            zeta_max_i = max(zeta_max_i, periodic_table.index(item[0]))

        self.zeta_max_i = zeta_max_i

    def min_alpha(self):
        '''
        To be used in configuration interaction to calculate the alpha parameter.
        We will be using:
        How to recover exponents: https://github.com/pyscf/pyscf/blob/f0fc18dc994e63e5dab132d7276eb22cdc9f25bf/pyscf/gto/mole.py#L3117
        Example on how to get exponent information: https://github.com/pyscf/pyscf/blob/master/examples/gto/11-basis_info.py
        
        What we are doing behind the scenes:
        # - For the correct basis in https://github.com/pyscf/pyscf/blob/master/pyscf/gto/basis/
        # - For the atoms in the molecule
        # - The first column of numbers indicates the exponent values (alphas).
        # Since we need an upper bound, we want the smallest of the exponents of the atoms of the molecule, in the right basis
        # eg alpha = min alphas
        '''

        pyscf_mol = self.molecule_data._pyscf_data['mol']

        alphas = [list(pyscf_mol.bas_exp(i)) for i in range(pyscf_mol.nbas)]
        alphas_func = lambda alphas: [item for sublist in alphas for item in sublist]

        self.alpha = np.min(alphas_func(alphas))
        return

    def save(self,json_name): 
        '''
        To save pyscf files: 
        https://github.com/pyscf/pyscf/blob/1de8c145abb3e1a7392df9118e8062e6fe6bde00/examples/ao2mo/01-outcore.py
        https://github.com/pyscf/pyscf/blob/1de8c145abb3e1a7392df9118e8062e6fe6bde00/examples/misc/02-chkfile.py
        
        To save MolecularData class:
        https://github.com/quantumlib/OpenFermion/blob/7c3581ad75716d1ff6a0043a516d271052a90e35/src/openfermion/chem/molecular_data.py#L567
        '''
        
        #The function takes MolecularData from file 'filename.hdf5' where filename is self.name
        self.molecule_data.save()

        molecule_properties = {}

        molecule_properties["N"] = self.N
        if hasattr(self, 'N_grid'): molecule_properties["N_grid"] = self.N_grid
        if hasattr(self, 'lambda_value'): molecule_properties["lambda_value"] = self.lambda_value
        if hasattr(self, 'lambda_value_grid'): molecule_properties["lambda_value_grid"] = self.lambda_value_grid
        if hasattr(self, 'lambda_value_low_rank'): molecule_properties["lambda_value_low_rank"] = self.lambda_value_low_rank
        if hasattr(self, 'Lambda_value'): molecule_properties["Lambda_value"] = self.Lambda_value
        if hasattr(self, 'Lambda_value_grid'): molecule_properties["Lambda_value_grid"] = self.Lambda_value_grid
        if hasattr(self, 'Gamma'): molecule_properties["Gamma"] = self.Gamma
        if hasattr(self, 'Gamma_grid'): molecule_properties["Gamma_grid"] = self.Gamma_grid
        if hasattr(self, 'eta'): molecule_properties["eta"] = self.eta
        if hasattr(self, 'Omega'): molecule_properties["Omega"] = self.Omega
        if hasattr(self, 'zeta_max_i'): molecule_properties["zeta_max_i"] = self.zeta_max_i

        if hasattr(self, 'alpha'): molecule_properties["alpha"] = self.alpha
        if hasattr(self, 'phi_max'): molecule_properties["phi_max"] = self.phi_max
        if hasattr(self, 'dphi_max'): molecule_properties["dphi_max"] = self.dphi_max
        if hasattr(self, 'grad_max'): molecule_properties["grad_max"] = self.grad_max
        if hasattr(self, 'hess_max'): molecule_properties["hess_max"] = self.hess_max
        if hasattr(self, 'lapl_max'): molecule_properties["lapl_max"] = self.lapl_max

        if hasattr(self, 'final_rank'): molecule_properties["final_rank"] = self.final_rank
        if hasattr(self, 'sparsity_d'): molecule_properties["sparsity_d"] = self.sparsity_d
        if hasattr(self, 'lambda_value_T'): molecule_properties["lambda_value_T"] = self.lambda_value_T
        if hasattr(self, 'lambda_value_U_V'): molecule_properties["lambda_value_U_V"] = self.lambda_value_U_V
        molecule_properties["xmax"] = self.xmax

        with open(json_name, "w") as fp:
            json.dump(molecule_properties,fp) 

    def load(self,json_name):
        '''
        To load MolecularData: https://github.com/quantumlib/OpenFermion/blob/7c3581ad75716d1ff6a0043a516d271052a90e35/src/openfermion/chem/molecular_data.py#L719
        '''

        try:
            with open(json_name, "r") as fp:
                molecule_properties = json.load(fp)

            if 'N' in molecule_properties.keys(): self.N = molecule_properties["N"]
            if 'N_grid' in molecule_properties.keys(): self.N_grid = molecule_properties["N_grid"]
            if 'lambda_value' in molecule_properties.keys(): self.lambda_value = molecule_properties["lambda_value"]
            if 'Lambda_value' in molecule_properties.keys(): self.Lambda_value = molecule_properties["Lambda_value"]
            if 'Gamma' in molecule_properties.keys(): self.Gamma = molecule_properties["Gamma"]
            if 'lambda_value_grid' in molecule_properties.keys(): self.lambda_value_grid = molecule_properties["lambda_value_grid"]
            if 'Lambda_value_grid' in molecule_properties.keys(): self.Lambda_value_grid = molecule_properties["Lambda_value_grid"]
            if 'lambda_value_low_rank' in molecule_properties.keys(): self.lambda_value_low_rank = molecule_properties["lambda_value_low_rank"]
            if 'Gamma_grid' in molecule_properties.keys(): self.Gamma_grid = molecule_properties["Gamma_grid"]
            if 'eta' in molecule_properties.keys(): self.eta = molecule_properties["eta"]
            if 'Omega' in molecule_properties.keys(): self.Omega = molecule_properties["Omega"]
            if 'zeta_max_i' in molecule_properties.keys(): self.zeta_max_i = molecule_properties["zeta_max_i"]

            if 'alpha' in molecule_properties.keys(): self.alpha = molecule_properties["alpha"]
            if 'phi_max' in molecule_properties.keys(): self.phi_max = molecule_properties["phi_max"]
            if 'dphi_max' in molecule_properties.keys(): self.dphi_max = molecule_properties["dphi_max"]
            if 'grad_max' in molecule_properties.keys(): self.grad_max = molecule_properties["grad_max"]
            if 'hess_max' in molecule_properties.keys(): self.hess_max = molecule_properties["hess_max"]
            if 'lapl_max' in molecule_properties.keys(): self.lapl_max = molecule_properties["lapl_max"]

            if 'final_rank' in molecule_properties.keys(): self.final_rank = molecule_properties["final_rank"]
            if 'sparsity_d' in molecule_properties.keys(): self.sparsity_d = molecule_properties["sparsity_d"]
            if 'lambda_value_T' in molecule_properties.keys(): self.lambda_value_T = molecule_properties["lambda_value_T"]
            if 'lambda_value_U_V' in molecule_properties.keys(): self.lambda_value_U_V = molecule_properties["lambda_value_U_V"]
            if 'xmax' in molecule_properties.keys(): self.xmax = molecule_properties["xmax"]

        except:
            pass

    def lambda_of_Hamiltonian_terms_2nd(self,grid, non_periodic = True, spinless = False):
        '''To be used in second quantization (interaction_picture) only'''

        V_dual = dual_basis_potential(grid = grid, spinless = spinless, non_periodic = non_periodic) # diagonal
        U_dual = dual_basis_external_potential(grid = grid, geometry = self.molecule_geometry, spinless = spinless, non_periodic = non_periodic) # diagonal

        T_primal = plane_wave_kinetic(grid, spinless = spinless) # diagonal
        
        Maj_op = get_majorana_operator(V_dual)
        l_maj = np.abs(np.array(list(Maj_op.terms.values())))
        lambda_V = sum(l_maj[1:]) # The first term is constant

        lambda_U = (U_dual.induced_norm() - U_dual.constant)/2 # division between 2 to take spin into account
        self.lambda_value_T = (T_primal.induced_norm() - T_primal.constant)/2 # division between 2 to take spin into account

        self.lambda_value_U_V = lambda_U+lambda_V

    def lambda_of_Hamiltonian_terms_1st(self, eta, Omega, N):
        '''To be used in first quantization'''
        
        def quadratic_sum(N): return N*(N+1)*(2*N + 1)**3
        
        sum_nu = quadratic_sum(int(N**(1/3)))

        lambda_U_V = (2*np.sqrt(3)*eta*(3*eta-1))*(N/Omega)**(1/3)

        lambda_T = eta/2 * (2*np.pi/Omega**(1/3))**2 * sum_nu

        return lambda_T, lambda_U_V


    def get_one_norm_int_woconst(self, one_body_integrals, two_body_integrals):
        """
        Returns 1-norm, emitting the constant term in the qubit Hamiltonian.
        See get_one_norm_int.

        Code mostly taken from https://github.com/quantumlib/OpenFermion/pull/725

        Parameters
        ----------
        one_body_integrals(ndarray) : An array of the one-electron integrals having
            shape of (n_orb, n_orb), where n_orb is the number of spatial orbitals.
        two_body_integrals(ndarray) : An array of the two-electron integrals having
            shape of (n_orb, n_orb, n_orb, n_orb).
        Returns
        -------
        one_norm : 1-Norm of the qubit Hamiltonian
        """

        n_orb = one_body_integrals.shape[0]

        htildepq = np.zeros(one_body_integrals.shape)
        for p in range(n_orb):
            for q in range(n_orb):
                htildepq[p, q] = one_body_integrals[p, q]
                for r in range(n_orb):
                    htildepq[p, q] += ((two_body_integrals[p, r, r, q]) -
                                    (1 / 2 * two_body_integrals[p, r, q, r]))

        one_norm = np.sum(np.absolute(htildepq))

        anti_sym_integrals = two_body_integrals - np.transpose(
            two_body_integrals, (0, 1, 3, 2))

        one_norm += 1 / 8 * np.sum(np.absolute(anti_sym_integrals))
        one_norm += 1 / 4 * np.sum(np.absolute(two_body_integrals))

        Lambda_value = max([1/2*np.max(np.absolute(htildepq)), 
                            1/16*np.max(np.absolute(anti_sym_integrals)), 
                            1/8*np.max(np.absolute(two_body_integrals))])

        Gamma_1bdy = np.count_nonzero(np.abs(htildepq)> tol)*2
        Gamma_2bdy = np.count_nonzero(np.absolute(two_body_integrals) > tol)
        Gamma_2bdy += np.count_nonzero(np.absolute(anti_sym_integrals) > tol)/2
        Gamma = Gamma_1bdy + Gamma_2bdy

        return one_norm, Lambda_value, Gamma    

    def spatial_from_spinorb(self, one_body_tensor, two_body_tensor):
        #Converting from spin orbitals to spatial orbitals
        n_spin_orbitals = one_body_tensor.shape[0]
        assert(one_body_tensor.shape[0] == two_body_tensor.shape[0])
        n_spatial_orbitals = n_spin_orbitals//2
        '''
        Example of the sumation that comes now
        a = np.arange(64)
        a = a.reshape(8,8) -> want to reshape to (4,4) summing by blocks of 2
        a = a.reshape(4,2,4,2).sum(axis = (1,3))
        '''

        # We add a 1/2 term because in spinorb_from_spatial each entry gets copied twice https://github.com/quantumlib/OpenFermion/blob/ce7b0023fea8721aee5796c82559254b3198d79d/src/openfermion/chem/molecular_data.py#L222-L260
        one_body_integrals = 1/2*one_body_tensor.reshape(n_spatial_orbitals,2,n_spatial_orbitals,2).sum(axis=(1,3))
        # We add a 1/4 term because in spinorb_from_spatial each entry gets copied four times https://github.com/quantumlib/OpenFermion/blob/ce7b0023fea8721aee5796c82559254b3198d79d/src/openfermion/chem/molecular_data.py#L222-L260
        two_body_integrals = 1/4*two_body_tensor.reshape(n_spatial_orbitals,2,n_spatial_orbitals,2,n_spatial_orbitals,2,n_spatial_orbitals,2).sum(axis = (1,3,5,7))

        return one_body_integrals, two_body_integrals

import numpy
import h5py
from itertools import combinations
class Molecule_Hamiltonian:

    def __init__(self, molecule_info, tools):

        self.molecule_info = molecule_info
        self.tools = tools

        # it is necessary to set to None to indicate to some methods that it is necessary to recalculate
        self.sparsity_d = None

        # set r value or final rank
        self.final_rank = 200 # set cholesky dimension
        self.N = 108 #todo IMPORTANT: 108 FOR REIHER, 152 FOR LI

        self.get_basic_parameters()


    # code extracted from https://doi.org/10.5281/zenodo.4248322
    def get_basic_parameters(self, molecular_hamiltonian=None):

        f = h5py.File(self.molecule_info+"eri_reiher.h5", "r")
        eri = f['eri'][()]
        h0 = f['h0'][()]
        f.close()

        f = h5py.File(self.molecule_info+"eri_reiher_cholesky.h5", "r")
        gval = f["gval"][()]
        gvec = f["gvec"][()]
        f.close()

        norb = h0.shape[1]
        nchol_max = gval.shape[0]
        thresh = 3.5e-5 # set threshold

        L = numpy.einsum("ij,j->ij",gvec,numpy.sqrt(gval))
        L = L.T.copy()
        L = L.reshape(nchol_max, norb, norb)

        T = h0 - 0.5 * numpy.einsum("pqqs->ps", eri, optimize=True) + numpy.einsum("pqrr->pq", eri, optimize = True)

        lambda_T = numpy.sum(numpy.abs(T))

        LR = L[:self.final_rank,:,:].copy()

        lambda_W = 0.25 * numpy.einsum("xij,xkl->",numpy.abs(LR), numpy.abs(LR), optimize=True)

        # save parameters to cost_methods
        self.lambda_value = lambda_T + lambda_W
        self.lambda_value_low_rank = self.lambda_value

        # Lambda_value is the max of all summed coefficients of T and LR
        V = 0.25 * numpy.einsum("xij,xkl->ijkl",numpy.abs(LR), numpy.abs(LR), optimize=True) 
        max_LR = max(numpy.abs(V).flatten())
        max_T = max(numpy.abs(T).flatten())

        self.Lambda_value = max(max_LR, max_T)

        # Gamma is the number of values over the threshold
        self.Gamma = np.count_nonzero( numpy.abs(T).flatten() >= thresh) + np.count_nonzero( numpy.abs(V).flatten() >= thresh)

        # number orbitals
        self.N = norb

    def low_rank_approximation(self, sparsify):

        return None, self.final_rank