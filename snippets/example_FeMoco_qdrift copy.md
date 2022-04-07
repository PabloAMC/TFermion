How to execute
=============

| Type  | Parameters | Description | Output |
|:-------------: |:-------------: |:-------------: |:-------------: |
| Execution example  | <table>  <thead>  <tr>  <th>molecule</th>  <th>method</th> </tr>  </thead>  <tbody>  <tr>  <td>FeMoco</td><td>Qdrift</td>  </tr>  <tr>  </tbody>  </table>     | How to execute one molecule (by Hamiltonian) with one method | T gate cost |

`Link to get FeMoco Hamiltonian files`: https://zenodo.org/record/4248322

Command
-------------
```
python main.py ./femoco/integrals/ qdrift
```

`FeMoco hamiltonian is in files eri_reiher.h5 and eri_reiher_cholesky.h5, which are inside integrals`

Output
---------

```
#####################################################################
##                             TFermion                            ##
##                                                                 ##
##  A non-Clifford gate cost assessment library of quantum phase   ##
##            estimation algorithms for quantum chemistry          ##
#####################################################################

<i> RESULT => The cost to calculate the energy of ./FEMOCO/INTEGRALS/ with method QDRIFT is 1.84e+23 T gates

** -------------------------------------------------- **
**                                                    **
** Execution time     => 0:00:04.454174  in hh:mm:ss  **
********************************************************

```