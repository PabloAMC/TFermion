# QPhase

QPhase is a repository that allows to calculate the expected cost T-gate cost of performing Phase Estimation with different methods.

Things that need to be done
1. Copy from QFold the code that allows to retrieve a given molecule from pubchem
2. Implement a code from OpenFermion that allows to calculate the constants required to calculate the T-gate cost.
  2.1 Most important variables are the number of orbitals (dependent on the basis) N and the sum of one and two body integrals $\lambda$
  2.2 Several basis are required, so we have to find a way to them.

Steps to install PSI4 + OpenFermion + OpenFermionPSI4 (from https://github.com/quantumlib/OpenFermion-Psi4/issues/44)

conda create -n openfermion python=3.6

conda activate openfermion

conda install -c psi4 psi4 as done here

pip install git+https://github.com/quantumlib/OpenFermion.git@master

pip install git+https://github.com/quantumlib/OpenFermion-Psi4.git@master

All this version are aligned.
