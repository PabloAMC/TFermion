<div align="center">    
 <img width="500" alt="T-Fermion" src="https://user-images.githubusercontent.com/20182937/138862534-b84836fb-9be1-4690-817b-766c68fc2d3e.png">   
</div>


<div align="center">
 [![Paper](http://img.shields.io/badge/arxiv-quant.ph.2110.05899-B31B1B.svg)](https://arxiv.org/abs/2110.05899)
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
<div align="center">    
<img width="100" alt="UCM-Logo" src="https://user-images.githubusercontent.com/20182937/138861126-660e548e-71fe-40ef-a8d3-f4385726f792.png">
</div>
