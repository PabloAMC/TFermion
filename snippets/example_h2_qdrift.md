How to execute
=============

| Type  | Parameters | Description | Output |
|:-------------: |:-------------: |:-------------: |:-------------: |
| Execution example  | <table>  <thead>  <tr>  <th>molecule</th>  <th>method</th>  <th>ao_labels</th>  </tr>  </thead>  <tbody>  <tr>  <td>H2</td><td>Qdrift</td><td>C 2p</td>  </tr>  <tr>  </tbody>  </table>     | How to execute one molecule (by name) with one method | T gate cost |

Command
-------------
```
python main.py h2 qdrift 'C 2p'
```

Output
---------

```
#####################################################################
##                             TFermion                            ##
##                                                                 ##
##  A non-Clifford gate cost assessment library of quantum phase   ##
##            estimation algorithms for quantum chemistry          ##
#####################################################################

<i> HF energy, MP2 energy, CCSD energy -1.094807962860512 -1.1152126817738932 -1.126778370572488
<i> RESULT => The cost to calculate the energy of H2 with method QDRIFT is 6.43e+16 T gates

** -------------------------------------------------- **
**                                                    **
** Execution time     => 0:00:02.460403  in hh:mm:ss  **
********************************************************


```