<div align="center">    
 <img width="500" alt="TFermion" src="https://user-images.githubusercontent.com/20182937/139062847-e45efa4c-b45c-4de4-9a0e-73bffbc2ff76.png">   
 
 [![Paper](http://img.shields.io/badge/arxiv-quant.ph:2110.05899-B31B1B.svg)](https://arxiv.org/abs/2110.05899)
 [![Paper](http://img.shields.io/badge/2022-Quantum-purple.svg)](https://quantum-journal.org/papers/q-2022-07-20-768/)
</div>
 
## Description   
TFermion is a library that allows to calculate the expected cost T-gate cost of performing Quantum Phase Estimation with different methods for arbitrary molecules and Hamiltonians. It uses methods from [OpenFermion](https://github.com/quantumlib/OpenFermion) and [Pyscf](https://github.com/pyscf/pyscf) and interfaces with [PubChem](https://pubchem.ncbi.nlm.nih.gov/) to download the corresponding molecular data. 

You can read our paper at [TFermion: A non-Clifford gate cost assessment library of quantum phase estimation algorithms for quantum chemistry](https://arxiv.org/abs/2110.05899).

The current implemented methods (we list the arxiv papers) are:
- `qdrift`, the qDRIFT method from [https://arxiv.org/abs/1811.08017](https://arxiv.org/abs/1811.08017).
- `rand_ham`, the Random Hamiltonian simulation method from [https://arxiv.org/abs/1811.08017](https://arxiv.org/abs/1811.08017).
- `taylor_naive`, the Naïve Taylorization method from from [https://arxiv.org/abs/1506.01020](https://arxiv.org/abs/1506.01020).
- `taylor_on_the_fly`, the Taylor on-the-fly method from [https://arxiv.org/abs/1506.01020](https://arxiv.org/abs/1506.01020).
- `configuration_interaction`, the method from [https://arxiv.org/abs/1506.01029](https://arxiv.org/abs/1506.01029).
- `low_depth_trotter`, the Trotterization method in plane waves from [https://arxiv.org/abs/1706.00023](https://arxiv.org/abs/1706.00023).
- `low_depth_taylor`, the Naïve Taylorization method in plane waves from [https://arxiv.org/abs/1706.00023](https://arxiv.org/abs/1706.00023).
- `low_depth_taylor_on_the_fly`, the on-the-fly Taylorization method in plane waves from [https://arxiv.org/abs/1706.00023](https://arxiv.org/abs/1706.00023).
- `shc_trotter`, similar to `low_depth_trotter` but using the SHC bound from [https://arxiv.org/abs/2107.07238](https://arxiv.org/abs/2107.07238), developing over the work of [https://arxiv.org/abs/2012.09194](https://arxiv.org/abs/2012.09194).
- `linear_t`, the method from [https://arxiv.org/abs/1805.03662](https://arxiv.org/abs/1805.03662).
- `sparsity_low_rank`, the method from [https://arxiv.org/abs/1902.02134](https://arxiv.org/abs/1902.02134).
- `interaction_picture`, the method from [https://arxiv.org/abs/1805.00675](https://arxiv.org/abs/1805.00675).

Other methods not yet implemented include:
- The double rank factorization from [https://arxiv.org/abs/2007.14460](https://arxiv.org/abs/2007.14460).
- The tensor hypercontraction method from [https://arxiv.org/abs/2011.03494](https://arxiv.org/abs/2011.03494).
- The qubitization and interaction picture methods from [https://arxiv.org/abs/1807.09802](https://arxiv.org/abs/1807.09802) and [https://arxiv.org/abs/2105.12767](https://arxiv.org/abs/2105.12767).
- Other Trotter methods with different bounds such as those from [https://arxiv.org/abs/2107.07238](https://arxiv.org/abs/2107.07238).

## How to install  
First, install dependencies   
```bash
# clone project   
git clone https://github.com/PabloAMC/TFermion

# install chemistry libraries
pip install openfermion pyscf

# install wrappers for chemistry libraries
pip install openfermionpsi4 openfermionpyscf

# install auxiliary libraries
pip install pandas numpy h5py scipy sympy matplotlib
 ```   

## How to run
```
python main.py [molecule name] [method name]
```

## Examples
You can find some snippet examples in the folder [snippets](/snippets)

## Contributing
We would love TFermion to become an useful tool for reseachers. If you want to join this mission, drop us a line at pabloamo@ucm.es.
Things that can be done include implementing new methods (see the snippet in the [snippet folder](/snippets)), or counting qubits. Also, some adaptation for periodic systems might be useful.


## Authorship
### Citation   
```
@article{casares2022tfermion,
  doi = {10.22331/q-2022-07-20-768},
  url = {https://doi.org/10.22331/q-2022-07-20-768},
  title = {{TF}ermion: {A} non-{C}lifford gate cost assessment library of quantum phase estimation algorithms for quantum chemistry},
  author = {Casares, Pablo A. M. and Campos, Roberto and Martin-Delgado, M. A.},
  journal = {{Quantum}},
  issn = {2521-327X},
  publisher = {{Verein zur F{\"{o}}rderung des Open Access Publizierens in den Quantenwissenschaften}},
  volume = {6},
  pages = {768},
  month = jul,
  year = {2022}
}
```   
### Contributors  
[P. A. M. Casares](https://github.com/PabloAMC) (Universidad Complutense de Madrid) and [R Campos](https://github.com/roberCo) (Universidad Complutense de Madrid).

### License
Apache 2.0 license.

<div align="center">
<img width="300" alt="UCM-Logo" src="https://user-images.githubusercontent.com/20182937/139064090-2f3ddc11-a140-44da-8339-0de2c86a6b7d.png">
</div>
