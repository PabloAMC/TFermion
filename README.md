<div align="center">    
 <img width="500" alt="T-Fermion" src="https://user-images.githubusercontent.com/20182937/139062847-e45efa4c-b45c-4de4-9a0e-73bffbc2ff76.png">   


 
 
 [![Paper](http://img.shields.io/badge/arxiv-quant.ph:2110.05899-B31B1B.svg)](https://arxiv.org/abs/2110.05899)
</div>
 
## Description   
T-Fermion is a library that allows to calculate the expected cost T-gate cost of performing Phase Estimation with different methods.

## How to install  
First, install dependencies   
```bash
# clone project   
git clone https://github.com/PabloAMC/TFermion

# install project
conda create -n tfermion python=3.6
conda activate tfermion
pip install pyscf
pip install openfermion
pip install openfermionpyscf
 ```   

## How to run
```
python main.py [molecule name] [method name]
```

### Citation   
```
@article{casares2021t,
  title={T-Fermion: A non-Clifford gate cost assessment library of quantum phase estimation algorithms for quantum chemistry},
  author={Casares, PAM and Campos, Roberto and Martin-Delgado, MA},
  journal={arXiv preprint arXiv:2110.05899},
  year={2021}
}
```   
### Contributors  
[P. A. M. Casares](https://github.com/PabloAMC) (Universidad Complutense de Madrid) and [R Campos](https://github.com/roberCo) (Universidad Complutense de Madrid).


<div align="center">

<img width="300" alt="UCM-Logo" src="https://user-images.githubusercontent.com/20182937/139063409-d33a6078-9218-4a8d-8016-78fb93c0b483.jpg">
</div>

