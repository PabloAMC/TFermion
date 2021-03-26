import openfermion

from openfermionpsi4 import run_psi4

from openfermion.utils import Grid
from openfermion.chem import geometry_from_pubchem, MolecularData
from openfermion.hamiltonians import plane_wave_hamiltonian
from openfermion.transforms  import  get_fermion_operator
from openfermion.transforms import jordan_wigner
from openfermion.circuits import low_rank_two_body_decomposition

import numpy as np
from pyscf import gto, scf, mcscf, fci, ao2mo
from pyscf.mcscf import avas

class Molecule:

    def __init__(self, name, tools):

        self.molecule_name = name
        self.tools = tools

        self.gamma_threshold = self.tools.config_variables['gamma_threshold']

        self.molecule_geometry = geometry_from_pubchem(self.molecule_name) #todo: do we prefer psi4 or pyscf? There are some functions in pyscf
        self.molecule_data = MolecularData(self.molecule_geometry, self.tools.config_variables['basis'], multiplicity = 1)
        self.molecule_psi4 = run_psi4(self.molecule_data,run_scf=True, run_mp2=True, run_fci=False)

        self.lambda_value, self.Lambda_value, self.N, self.Gamma = self.get_parameters(self.molecule_psi4)

    def get_parameters(self, molecule):

        lambda_value_one_body, Lambda_value_one_body, N_one_body, gamma_one_body = self.get_parameters_from_row(self.molecule_psi4.one_body_integrals, 0, -1, 0, 0)
        lambda_value_two_body, Lambda_value_two_body, N_two_body, gamma_two_body = self.get_parameters_from_row(self.molecule_psi4.two_body_integrals, 0, -1, 0, 0)

        assert N_one_body == N_two_body

        return lambda_value_one_body+lambda_value_two_body, max(Lambda_value_one_body, Lambda_value_two_body), N_one_body, gamma_one_body+gamma_two_body

    def build_grid(self):
        grid = Grid(dimensions = 3, length = 5, scale = 1.) # La complejidad
        plane_wave_H = plane_wave_hamiltonian(grid, self.molecule_geometry, True)

    # recursive method that iterates over all rows of a molecule to get the parameters:
    # lambda_value is the sum all coefficients of the hamiltonian (sum of all terms)
    # Lambda_value is the maximum value of all terms
    # N is the number of orbitals
    # gamma is the total number of elements (without counting values under some threshold)
    def get_parameters_from_row(self, row, lambda_value, Lambda_value, N, gamma):

        for r in row:

            if type(r) == np.ndarray:
                lambda_value, Lambda_value, N, gamma = self.get_parameters_from_row(r, lambda_value, Lambda_value, N, gamma)
            else:

                # lambda_value is the sum of all terms of all rows
                lambda_value += abs(r)
                
                # Lambda_value is the maximum of all terms of all rows
                if abs(r) > Lambda_value: Lambda_value = abs(r)

                # N is the number of orbitals and it should be equal in all rows (if it is 0, initialize it)
                if N == 0: N = len(row) 
                else: assert N == len(row)

                # gamma is the count of elements above a threshold (threshold close to 0 or 0 to delete small terms)
                if abs(r) > self.gamma_threshold: gamma += 1

        return lambda_value, Lambda_value, N, gamma

    def molecular_orbital_parameters(self):
        '''
        Aim: caluclate phi_max and dphi_max
        - To calculate the ao basis. Bibliography https://onlinelibrary.wiley.com/doi/pdf/10.1002/wcms.1123?casa_token=M0hDMDgf0VkAAAAA:qOQVt0GDe2TD7WzAsoHCq0kLzNgAQFjssF57dydp1rsr4ExjZ1MEP75eD4tkjpATrpkd81qnWjJmrA
        https://github.com/pyscf/pyscf-doc/blob/93f34be682adf516a692e28787c19f10cbb4b969/examples/gto/11-basis_info.py
        Some useful methods from mol class (can be found using dir(mol))
        'bas_ctr_coeff',
        'bas_exp',
        'bas_kappa'
        To use them use, for example, check the documentation with help(mol.bas_ctr_coeff)

        - For conversion of ao to mo
        https://github.com/pyscf/pyscf/tree/5796d1727808c4ab6444c9af1f8af1fad1bed450/pyscf/ao2mo
        https://github.com/pyscf/pyscf-doc/tree/93f34be682adf516a692e28787c19f10cbb4b969/examples/ao2mo
        
        General Integral transformation module
        ======================================
        Simple usage::
            >>> from pyscf import gto, scf, ao2mo
            >>> mol = gto.M(atom='H 0 0 0; F 0 0 1')
            >>> mf = scf.RHF(mol).run() # mf.mo_coeff contains the basis change matrix from ao to mo
            >>> mo_ints = ao2mo.kernel(mol, mf.mo_coeff) # Molecular integrals (not interested directly con them)

        - For active space, use something like
            >>> mf = scf.ROHF(mol)
            >>> mf.kernel()
            >>> from pyscf.mcscf import avas
            >>> norb, ne_act, orbs = avas.avas(mf, ao_labels, canonicalize=False)

            and orbs will have the coefficients of the molecular orbitals

        '''
        return


    def active_space(self, ao_labels):
        '''
        Inputs:
        ao_labels: list #atomic orbitals needed to construct the active space EXAMPLE: ao_labels = ['Fe 3d', 'C 2pz']
        ss: list #list of spins
        nroots: list # number of states to be solved for each fci? solver

        What interests us from here are orbitals and the hamiltonian

        Example taken from https://github.com/pyscf/pyscf-doc/blob/93f34be682adf516a692e28787c19f10cbb4b969/examples/mcscf/43-avas.py
        Documentation reference: https://github.com/pyscf/pyscf/blob/5796d1727808c4ab6444c9af1f8af1fad1bed450/pyscf/mcscf/avas.py
        WARNING: LARGE MOLECULES (~20 ATOMS) TAKES A HUGE AMOUNT OF TIME (3-4 HOURS EASILY) EVEN IN THE SELF-CONSISTENT FIELD CALCULATION
        '''

        # Selecting the active space
        # FOR PYSCF COMPUTATIONS
        self.mol = gto.Mole()
        self.mol = gto.M(
            atom = self.molecule_geometry,
            basis = self.tools.config_variables['basis'])
        self.myhf = scf.RHF(mol) #.x2c() The x2c is relativistic. We are not so desperate :P
        self.myhf.kernel() # WARNING: LARGE MOLECULES CAN TAKE IN THE ORDER OF HOURS TO COMPUTE THIS

        norb, ne_act, orbs = avas.avas(myhf, ao_labels, canonicalize=False)
        # norb is number of orbitals
        # ne_act is number of active electrons
        # orbs is the mo_coeff, that is, the change of basis matrix from atomic orbitals -> molecular orbitals
        mo_ints = ao2mo.kernel(mol, orbs) # Molecular integrals h_{ijkl} appearing in the Hamiltonian
        #todo: unclear how to separate from here the active Hamiltonian, which is what we care about

    def low_rank_approximation(self):
        '''
        Aim: get a low rank (rank-truncated) hamiltonian such that the error using say mp2 is smaller than chemical accuracy. Then use that Hamiltonian to compute the usual terms
        Perform low rank approximation using
        https://github.com/quantumlib/OpenFermion/blob/4781602e094699f0fe0844bcded8ef0d45653e81/src/openfermion/circuits/low_rank.py#L76
        and see how precise that is using MP2 
        https://github.com/psi4/psi4numpy
        See also a discussion on this topic: https://github.com/quantumlib/OpenFermion/issues/708
        Costumizing Hamiltonian: https://github.com/pyscf/pyscf-doc/blob/master/examples/scf/40-customizing_hamiltonian.py    
        '''
        # Iterate to find the maximum truncation error that induces the smallest error

        # electronic repulsion integrals
        eri_4fold = ao2mo.kernel(self.mol, self.myhf.mo_coeff, compact=False)
        eri_shape = eri_4fold.shape
        #Reshape into 4 indices matrix
        two_body_coefficients = eri_4fold.reshape(np.array([int(np.sqrt(eri_shape[0]))]*2 + [int(np.sqrt(eri_shape[1]))]*2))#todo: Is this the correct ordering???

        truncation_threshold = 1e-8
        
        lambda_ls, one_body_squares, one_body_correction, truncation_value = low_rank_two_body_decomposition(two_body_coefficients,
                                                                                                            truncation_threshold=trunctation_threshold,
                                                                                                            final_rank=None,
                                                                                                            spin_basis=True)

        mol = gto.M()
        n = 10
        mol.nelectron = #todo 

        mf = scf.RHF(mol)

        mf.get_hcore = lambda *args: #todo this does not change, except the term that is added
        mf.get_ovlp = lambda *args: #todo this should not change???
        # ao2mo.restore(8, eri, n) to get 8-fold permutation symmetry of the integrals
        # ._eri only supports the two-electron integrals in 4-fold or 8-fold symmetry.
        eri = #todo
        mf._eri = ao2mo.restore(8, eri, mol.nelectron)
        mol.incore_anyway = True


        return