How to execute
=============

| Type  | Parameters | Description | Output |
|:-------------: |:-------------: |:-------------: |:-------------: |
| Execution example  | <table>  <thead>  <tr>  <th>molecule</th>  <th>method</th> </tr>  </thead>  <tbody>  <tr>  <td>FeMoco</td><td>Qdrift</td>  </tr>  <tr>  </tbody>  </table>     | How to execute one molecule (by Hamiltonian) with one method | T gate cost |

`Link to get FeMoco Hamiltonian files`: https://zenodo.org/record/4248322

Command
-------------
Create a folder inside TFermion folder named '/integrals' and put 'eri_reiher.h5' and 'eri_reiher_cholesky.h5' in it. Similarly for 'eri_li.h5 and 'eri_li_cholesky.h5'.

```
python main.py 'integrals/eri_reiher' sparsity_low_rank
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

Backend MacOSX is interactive backend. Turning interactive mode on.
<i> RESULT => The cost to calculate the energy of INTEGRALS/ERI_REIHER with method SPARSITY_LOW_RANK is 8.11e+12 T gates

** -------------------------------------------------- **
**                                                    **
** Execution time     => 0:00:14.631222  in hh:mm:ss  **
********************************************************
```