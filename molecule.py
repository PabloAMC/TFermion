import itertools
import openfermion

from openfermionpsi4 import run_psi4
from openfermionpyscf import run_pyscf
from openfermionpyscf._run_pyscf import prepare_pyscf_molecule, compute_integrals, compute_scf

from openfermion.utils import Grid
import openfermion.ops.representations as reps
from openfermion.chem import geometry_from_pubchem, MolecularData
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.hamiltonians import plane_wave_hamiltonian
from openfermion.transforms  import  get_fermion_operator
from openfermion.transforms import jordan_wigner
from openfermion.circuits import low_rank_two_body_decomposition

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

class Molecule:

    def __init__(self, name, tools, program = 'pyscf'):

        self.molecule_name = name
        self.tools = tools
        self.program = program

        self.gamma_threshold = self.tools.config_variables['gamma_threshold']
        self.molecule_geometry = geometry_from_pubchem(self.molecule_name) #todo: do we prefer psi4 or pyscf? There are some functions in pyscf
        
        #From OpenFermion
        self.molecule_data = MolecularData(self.molecule_geometry, self.tools.config_variables['basis'], multiplicity = 1)

        #todo: add possibility of boundary conditions https://sunqm.github.io/pyscf/tutorial.html#initializing-a-crystal
        if program == 'psi4': 
            self.molecule_psi4 = run_psi4(self.molecule_data,run_scf=True, run_mp2=False, run_fci=False)

        elif program == 'pyscf':
            self.molecule_pyscf = run_pyscf(self.molecule_data,run_scf=True, run_mp2=False, run_fci=False)
        
        self.N = self.molecule_data.n_orbitals * 2 # The 2 is due to orbitals -> spin orbitals
        self.get_basic_parameters()
        '''            
        self.molecule_pyscf = gto.Mole()
        self.molecule_pyscf = gto.M(
            atom = self.molecule_geometry,
            basis = self.tools.config_variables['basis'])
        self.myhf = scf.RHF(self.molecule_pyscf) #.x2c() The x2c is relativistic. We are not so desperate :P
        self.myhf.kernel() # WARNING: LARGE MOLECULES CAN TAKE IN THE ORDER OF HOURS TO COMPUTE THIS
        '''

    def get_basic_parameters(self, threshold = 0, occupied_indices = None, active_indices = None):

        molecular_hamiltonian = self.molecule_data.get_molecular_hamiltonian(occupied_indices=occupied_indices, active_indices=active_indices)
        fermion_operator = openfermion.get_fermion_operator(molecular_hamiltonian)
        JW_op = openfermion.transforms.jordan_wigner(fermion_operator)
        #BK_op = openfermion.transforms.bravyi_kitaev(fermion_operator) #Results seem to be the same no matter what transform one uses

        d = JW_op.terms
        del d[()]
        l = abs(np.array(list(d.values())))
        self.lambd = sum(l)
        self.Lambda = max(l)
        self.Gamma = np.count_nonzero(l >= threshold)

    def build_grid(self):
        grid = Grid(dimensions = 3, length = 5, scale = 1.) # La complejidad
        plane_wave_H = plane_wave_hamiltonian(grid, self.molecule_geometry, True)

        # recursive method that iterates over all rows of a molecule to get the parameters:
        # lambda_value is the sum all coefficients of the hamiltonian (sum of all terms)
        # Lambda_value is the maximum value of all terms
        # N is the number of orbitals
        # gamma is the total number of elements (without counting values under some threshold)

    def active_space(self, ao_labels):
        '''
        Inputs:
        ao_labels: list #atomic orbitals needed to construct the active space. EXAMPLE: ao_labels = ['Fe 3d', 'C 2pz']

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

        # Selecting the active space
        pyscf_scf = self.molecule_pyscf._pyscf_data['scf'] #similar to https://github.com/quantumlib/OpenFermion-PySCF/blob/8b8de945db41db2b39d588ff0396a93566855247/openfermionpyscf/_pyscf_molecular_data.py#L47
        ncas, ne_act_cas, mo_coeff, (n_mocore, n_mocas, n_movir) = avas.avas(pyscf_scf, ao_labels, canonicalize=False)
        # IMPORTANT: Line 191 from avas.py now reads. Modify it 
        #    return ncas, nelecas, mo, (mocore.shape[1], mocas.shape[1], movir.shape[1])

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
        #todo: check whether we want natural orbitals or not
        pyscf_mcscf = mcscf.CASSCF(pyscf_scf, ncas, ne_act_cas).run(mo_coeff) #Inspired by the mini-example in avas documentation link above

        self.molecule_data._pyscf_data['mcscf'] = pyscf_mcscf
        self.molecule_data.mcscf_energy = pyscf_mcscf.e_tot

        self.molecule_data.orbital_energies = pyscf_mcscf.mo_energy.astype(float)
        self.molecule_data.canonical_orbitals = pyscf_mcscf.mo_coeff.astype(float)

        return (ne_act_cas, n_mocore, n_mocas, n_movir)

    def low_rank_approximation(self, occupied_indices = [], active_indices = [], virtual_indices = []):
        '''
        Aim: get a low rank (rank-truncated) hamiltonian such that the error using say mp2 is smaller than chemical accuracy. Then use that Hamiltonian to compute the usual terms
        
        Args: 
        occupied_indices: list = []
        active_indices: list = []
        virtual_indices: list = []

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

        Probably beyond our interest: (if we wanted to create a pyscf_mol object with the truncated Hamiltonian, which we have skipped over)
        One can truncate the basis using something similar to https://github.com/pyscf/pyscf/blob/9c8b06d481623b50ccdca9f88d833de7320ac3cd/examples/gto/04-input_basis.py#L98
        To select get the ao_labels: https://github.com/pyscf/pyscf/blob/5796d1727808c4ab6444c9af1f8af1fad1bed450/pyscf/gto/mole.py#L1526
        To get the overlap matrix https://github.com/pyscf/pyscf/blob/5796d1727808c4ab6444c9af1f8af1fad1bed450/pyscf/mcscf/avas.py#L128
        '''

        CHEMICAL_ACCURACY = 0.0015 #in Hartrees, according to http://greif.geo.berkeley.edu/~driver/conversions.html

        pyscf_scf = self.molecule_data._pyscf_data['scf']
        pyscf_mol = self.molecule_data._pyscf_data['mol']
        two_body_integrals = self.molecule_data.two_body_integrals         # electronic repulsion integrals

        if occupied_indices or virtual_indices:
            new_core_constant, new_one_body_integrals, new_two_body_integrals = self.molecule_data.get_active_space_integrals(occupied_indices = occupied_indices, active_indices = active_indices)
        else:
            new_core_constant = 0
            new_two_body_integrals = two_body_integrals
            new_one_body_integrals = self.molecule_data.one_body_integrals

        # From here------------------------------------------------

        def low_rank_truncation_mp2_energy(threshold):
        
            print('<i> Trucation threshold =', threshold)

            lambda_ls, one_body_squares, one_body_correction, truncation_value = low_rank_two_body_decomposition(two_body_integrals,
                                                                                                    truncation_threshold=threshold,
                                                                                                    final_rank=None,
                                                                                                    spin_basis=False)

            print('<i> Rank =', len(lambda_ls))

            # Electronic Repulsion Integral #todo: does not get the right symmetry
            eri = np.einsum('l,lpq,lrs->pqrs',lambda_ls, (one_body_squares + np.transpose(one_body_squares, (0,2,1)))/2, (one_body_squares + np.transpose(one_body_squares, (0,2,1)))/2)

            # Integrals have type complex but they do not have imaginary part
            eri = np.real_if_close(eri)
            assert(np.isreal(eri).all())

            #Converting from spin orbitals to spatial orbitals
            n_spin_orbitals = eri.shape[0]
            n_spatial_orbitals = n_spin_orbitals//2
            '''
            Example of the sumation that comes now
            a = np.arange(64)
            a = a.reshape(8,8) -> want to reshape to (4,4) summing by blocks of 2
            a = a.reshape(4,2,4,2).sum(axis = (1,3))
            '''
            one_body_correction = one_body_correction.reshape(n_spatial_orbitals,2,n_spatial_orbitals,2).sum(axis=(1,3))
            eri = eri.reshape(n_spatial_orbitals,2,n_spatial_orbitals,2,n_spatial_orbitals,2,n_spatial_orbitals,2).sum(axis = (1,3,5,7))
            
            # Checking some of the symmetries
            assert(np.isclose(eri, np.transpose(eri, (3,2,1,0)), rtol = 1).all() and np.isclose(eri, np.transpose(eri, (2,3,0,1)), rtol = 1).all())

            mol = gto.M()
            mol.nelectron = self.molecule_pyscf.n_electrons

            mf = scf.RHF(mol)

            mf.get_hcore = lambda *args: pyscf_scf.get_hcore() + one_body_correction # this does not change, except the term that is added
            mf.get_ovlp = lambda *args: pyscf_scf.get_ovlp()
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
        exact_E = low_rank_truncation_mp2_energy(threshold = 0)

        nconstraint = scipy.optimize.NonlinearConstraint(fun = lambda threshold: low_rank_truncation_mp2_energy(threshold) - exact_E, lb = -CHEMICAL_ACCURACY, ub = +CHEMICAL_ACCURACY, keep_feasible = True)
        lconstraint = scipy.optimize.LinearConstraint(A = np.array([1]), lb = 1e-10, ub = 1, keep_feasible = True)
        result = scipy.optimize.minimize(fun = lambda threshold: 1e-2/(threshold+1e-4), x0 = 1e-4, constraints = [nconstraint, lconstraint], tol = .01*CHEMICAL_ACCURACY, options = {'maxiter': 50}, method='COBYLA') # Works with COBYLA, but not with SLSQP (misses the boundaries) or trust-constr (oscillates)
        threshold = float(result['x'])
        approximate_E = low_rank_truncation_mp2_energy(threshold = threshold)
        print('<i> MP2 energy error in the truncation', approximate_E-exact_E)

        lambda_ls, one_body_squares, one_body_correction, truncation_value = low_rank_two_body_decomposition(new_two_body_integrals,
                                                                                                            truncation_threshold=threshold,
                                                                                                            final_rank=None,
                                                                                                            spin_basis=False)

        final_rank = len(lambda_ls)

        two_body_coefficients = np.einsum('l,lpr,lqs->pqrs',lambda_ls, one_body_squares, one_body_squares)

        one_body_coefficients, _ = spinorb_from_spatial(new_one_body_integrals, new_two_body_integrals)
        one_body_coefficients = one_body_correction + one_body_coefficients

        constant = new_core_constant+self.molecule_data.nuclear_repulsion

        #todo: the 1/2 term in front of the two_body_coefficients should be there? -> Check the definition of get_molecular_hamiltonian https://github.com/quantumlib/OpenFermion/blob/40f4dd293d3ac7759e39b0d4c061b391e9663246/src/openfermion/chem/molecular_data.py#L878
        molecular_hamiltonian = reps.InteractionOperator(constant, one_body_coefficients, 1/2 * two_body_coefficients)

        return molecular_hamiltonian, final_rank

    def molecular_orbital_parameters(self):
        '''
        Aim: caluclate phi_max and dphi_max
        - To calculate the ao basis. Bibliography https://onlinelibrary.wiley.com/doi/pdf/10.1002/wcms.1123?casa_token=M0hDMDgf0VkAAAAA:qOQVt0GDe2TD7WzAsoHCq0kLzNgAQFjssF57dydp1rsr4ExjZ1MEP75eD4tkjpATrpkd81qnWjJmrA

        - COMPUTE MO VALUES AND THEIR DERIVATIVES IN THE SPACE: https://github.com/pyscf/pyscf/blob/master/examples/gto/24-ao_value_on_grid.py (It's totally awesome that this exists)

        - For conversion of ao to mo
        https://github.com/pyscf/pyscf/tree/5796d1727808c4ab6444c9af1f8af1fad1bed450/pyscf/ao2mo
        https://github.com/pyscf/pyscf-doc/tree/93f34be682adf516a692e28787c19f10cbb4b969/examples/ao2mo
        
        Procedure
        We evaluate the molecular orbital functions and their derivatives in random points and return the highest value
        '''

        pyscf_scf = self.molecule_data._pyscf_data['scf']
        pyscf_mol = self.molecule_data._pyscf_data['mol']

        #todo: how to choose the coordinates to fit the molecule well? -> Use geometry from the molecule
        #coords = np.random.random((100,3)) # todo: dumb, to be substituted

        coords = np.empty((0, 3))
        for _, at_coord in pyscf_mol.geometry:
            coord = np.vstack((coord, np.array(at_coord) + 10 * BOHR * np.random.random((50,3)))) # Random coords around the atomic positions

        # deriv=1: orbital value + gradients 
        ao_p = pyscf_mol.eval_gto('GTOval_sph_deriv1', coords)  # (4,Ngrids,n) array
        ao = ao_p[0]
        ao_grad = ao_p[1:4]  # x, y, z

        mo = ao.dot(pyscf_scf.mo_coeff)
        mo_grad = [x.dot(pyscf_scf.mo_coeff) for x in ao_grad]

        phi_max = np.max(mo)
        dphi_max = np.max(mo_grad)

        return phi_max, dphi_max