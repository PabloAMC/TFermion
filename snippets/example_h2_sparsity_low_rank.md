How to execute
=============

| Type  | Parameters | Description | Output |
|:-------------: |:-------------: |:-------------: |:-------------: |
| Execution example  | <table>  <thead>  <tr>  <th>molecule</th>  <th>method</th>  <th>ao_labels</th>  </tr>  </thead>  <tbody>  <tr>  <td>H2</td><td>Sparsity Low Rank</td><td>H 1s</td>  </tr>  <tr>  </tbody>  </table>     | How to execute one molecule (by name) with one method | T gate cost |

Command
-------------
```
python main.py h2 sparsity_low_rank 'H 1s'
```

`'H 1s'` is an optional parameter. If not included, the method will be executed with the active space being the whole space.

Output
---------

```
#####################################################################
##                             TFermion                            ##
##                                                                 ##
##  A non-Clifford gate cost assessment library of quantum phase   ##
##            estimation algorithms for quantum chemistry          ##
#####################################################################

<i> HF energy, MP2 energy, CCSD energy -1.0948079628605116 -1.1152126817738928 -1.126778370572344
CASSCF energy = -1.12352494971892
CASCI E = -1.12352494971892  E(CI) = -1.65270216063892  S^2 = 0.0000000
<i> RESULT => The cost to calculate the energy of H2 with method SPARSITY_LOW_RANK is 8.76e+09 T gates

** -------------------------------------------------- **
**                                                    **
** Execution time     => 0:00:02.770030  in hh:mm:ss  **
********************************************************
```
