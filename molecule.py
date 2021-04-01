import openfermion

from openfermionpsi4 import run_psi4
from openfermionpyscf import run_pyscf, prepare_pyscf_molecule, compute_scf, compute_integrals

from openfermion.utils import Grid
from openfermion.chem import geometry_from_pubchem, MolecularData
from openfermion.hamiltonians import plane_wave_hamiltonian
from openfermion.transforms  import  get_fermion_operator
from openfermion.transforms import jordan_wigner
from openfermion.circuits import low_rank_two_body_decomposition

import numpy as np
import time
from pyscf import gto, scf, mcscf, fci, ao2mo
from pyscf.mcscf import avas

class Molecule:

    def __init__(self, name, tools, program = 'pyscf'):

        self.molecule_name = name
        self.tools = tools
        self.program = program

        self.gamma_threshold = self.tools.config_variables['gamma_threshold']
        self.molecule_geometry = geometry_from_pubchem(self.molecule_name) #todo: do we prefer psi4 or pyscf? There are some functions in pyscf
        
        #From OpenFermion
        self.molecule_data = MolecularData(self.molecule_geometry, self.tools.config_variables['basis'], multiplicity = 1)

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

        Objects we use
        molecule_data: MolecularData https://quantumai.google/reference/python/openfermion/chem/MolecularData
        molecule_pyscf: PyscfMolecularData https://github.com/quantumlib/OpenFermion-PySCF/blob/8b8de945db41db2b39d588ff0396a93566855247/openfermionpyscf/_pyscf_molecular_data.py#L23
        _ : A pyscf molecule instance https://github.com/pyscf/pyscf/blob/master/pyscf/gto/mole.py
        scf: scf method https://github.com/pyscf/pyscf/blob/7be5e015b2b40181755c71d888449db936604660/pyscf/scf/__init__.py#L123
        mcscf: mcscf method https://github.com/pyscf/pyscf/blob/7be5e015b2b40181755c71d888449db936604660/pyscf/mcscf/__init__.py#L193
        
        Returns:
        - occupied_indices
        - active_indices
        These indices can be used in self.get_basic_parameters(). 
        Also modifies self.molecule_data and self.molecule_pyscf in place.
        '''

        #todo: I don't like the idea of accessing private methods, but I see no other way
        # Selecting the active space
        scf = self.molecule_pyscf._pyscf_data.get('scf', None) #similar to https://github.com/quantumlib/OpenFermion-PySCF/blob/8b8de945db41db2b39d588ff0396a93566855247/openfermionpyscf/_pyscf_molecular_data.py#L47
        ncas, ne_act_cas, mo_coeff = avas.avas(scf, ao_labels, canonicalize=False)
        #todo: avas should also return mocore.shape[1] and movir.shape[1]: https://github.com/pyscf/pyscf/blob/5796d1727808c4ab6444c9af1f8af1fad1bed450/pyscf/mcscf/avas.py#L165
        scf.mo_coeff = mo_coeff

        #todo: should we modify the mo_coeff of the scf? If so should we modify them again after the mcscf calculation?
        # Correcting molecular coefficients 
        self.molecule_data.canonical_orbitals = mo_coeff.astype(float)
        self.molecule_pyscf._canonical_orbitals = mo_coeff.astype(float)
        self.molecule_data._pyscf_data['scf'] = scf

        # Get two electron integrals
        one_body_integrals, two_body_integrals = compute_integrals(self.molecule_pyscf, scf)
        self.molecule_data.one_body_integrals = one_body_integrals
        self.molecule_data.two_body_integrals = two_body_integrals

        self.molecule_data.overlap_integrals = scf.get_ovlp()

        # This does not give the natural orbitals. If those are wanted check https://github.com/pyscf/pyscf/blob/7be5e015b2b40181755c71d888449db936604660/pyscf/mcscf/__init__.py#L172
        # Complete Active Space Self Consistent Field (CASSCF), an option of Multi-Configuration Self Consistent Field (MCSCF) calculation. A more expensive alternative would be Complete Active Space Configuration Interaction (CASCI)
        #todo: check whether we want natural orbitals or not
        mcscf = mcscf.CASSCF(scf, ncas, ne_act_cas).run(mo_coeff) #Inspired by the mini-example in avas documentation link above

        self.molecule_data._pyscf_data['mcscf'] = mcscf
        self.molecule_data.mcscf_energy = mcscf.e_tot

        self.molecule_data.orbital_energies = mcscf.mo_energy.astype(float)
        self.molecule_data.molecule.canonical_orbitals = mcscf.mo_coeff.astype(float)

        #todo: return also the active and occupied indices.
        return

    def low_rank_approximation(self):
        '''
        Aim: get a low rank (rank-truncated) hamiltonian such that the error using say mp2 is smaller than chemical accuracy. Then use that Hamiltonian to compute the usual terms
        Perform low rank approximation using
        https://github.com/quantumlib/OpenFermion/blob/4781602e094699f0fe0844bcded8ef0d45653e81/src/openfermion/circuits/low_rank.py#L76
        and see how precise that is using MP2: 
        https://github.com/psi4/psi4numpy, https://github.com/pyscf/pyscf/blob/c9aa2be600d75a97410c3203abf35046af8ca615/pyscf/mp/mp2.py#L411
        See also a discussion on this topic: https://github.com/quantumlib/OpenFermion/issues/708
        Costumizing Hamiltonian: https://github.com/pyscf/pyscf-doc/blob/master/examples/scf/40-customizing_hamiltonian.py    
        '''
        # Iterate to find the maximum truncation error that induces the smallest error

        # electronic repulsion integrals
        eri_4fold = ao2mo.kernel(self.mol, self.myhf.mo_coeff, compact=False)
        eri_shape = eri_4fold.shape
        #Reshape into 4 indices matrix
        two_body_coefficients = eri_4fold.reshape(np.array([int(np.sqrt(eri_shape[0]))]*2 + [int(np.sqrt(eri_shape[1]))]*2))#todo: Is this the correct ordering???


        # From here------------------------------------------------
        truncation_threshold = 1e-8
        
        lambda_ls, one_body_squares, one_body_correction, truncation_value = low_rank_two_body_decomposition(two_body_coefficients,
                                                                                                            truncation_threshold=trunctation_threshold,
                                                                                                            final_rank=None,
                                                                                                            spin_basis=True)

        eri = np.einsum('l,pql,rsl->pqrs',lambda_ls, one_body_squares, one_body_squares) # Is order right? #todo: multiply by lambda_ls

        mol = gto.M()
        mol.nelectron = self.molecule_pyscf.nelect

        mf = scf.RHF(mol)

        mf.get_hcore = lambda *args: self.myhf.get_hcore + one_body_correction #todo this does not change, except the term that is added
        mf.get_ovlp = lambda *args: self.myhf.get_ovlp #todo this should not change???
        # ao2mo.restore(8, eri, n) to get 8-fold permutation symmetry of the integrals
        # ._eri only supports the two-electron integrals in 4-fold or 8-fold symmetry.
        
        mf._eri = ao2mo.restore(8, eri, mol.nelectron)
        mol.incore_anyway = True

        #todo: frozen_orbitals
        pt = mf.MP2().set(frozen = frozen_orbitals).run()
        energy = pt.tot_energy
        # Until here------------------------------------ Iterate to see how high can we put the threshold without damaging the energy estimates (error up to chemical precision)

        return


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


