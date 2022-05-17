How to execute
=============

| Type  | Parameters | Description | Output |
|:-------------: |:-------------: |:-------------: |:-------------: |
| Execution example  | <table>  <thead>  <tr>  <th>molecule</th>  <th>method</th>  <th>ao_labels</th>  </tr>  </thead>  <tbody>  <tr>  <td>H2</td><td>All available methods</td><td>H 1s</td>  </tr>  <tr>  </tbody>  </table>     | How to execute one molecule (by name) with all available methods | T gate cost |

Command
-------------
```
python main.py h2 all 'H 1s'
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

<i> HF energy, MP2 energy, CCSD energy -1.0948079628605116 -1.1152126817738928 -1.126778370572339
<i> RESULT => The cost to calculate the energy of H2 with method QDRIFT is 6.24e+16 T gates
<i> RESULT => The cost to calculate the energy of H2 with method RAND_HAM is 3.04e+17 T gates
<i> RESULT => The cost to calculate the energy of H2 with method TAYLOR_NAIVE is 4.13e+13 T gates
<i> RESULT => The cost to calculate the energy of H2 with method TAYLOR_ON_THE_FLY is 2.21e+27 T gates
<i> RESULT => The cost to calculate the energy of H2 with method CONFIGURATION_INTERACTION is 4.70e+36 T gates
<i> RESULT => The cost to calculate the energy of H2 with method LOW_DEPTH_TROTTER is 7.10e+22 T gates
<i> RESULT => The cost to calculate the energy of H2 with method SHC_TROTTER is 1.37e+22 T gates
<i> RESULT => The cost to calculate the energy of H2 with method LOW_DEPTH_TAYLOR is 2.18e+15 T gates
<i> RESULT => The cost to calculate the energy of H2 with method LOW_DEPTH_TAYLOR_ON_THE_FLY is 8.75e+22 T gates
<i> RESULT => The cost to calculate the energy of H2 with method LINEAR_T is 2.70e+13 T gates
<i> RESULT => The cost to calculate the energy of H2 with method SPARSITY_LOW_RANK is 8.65e+09 T gates
<i> RESULT => The cost to calculate the energy of H2 with method INTERACTION_PICTURE is 2.10e+18 T gates

** -------------------------------------------------- **
**                                                    **
** Execution time     => 0:00:39.631718  in hh:mm:ss  **
********************************************************
```
