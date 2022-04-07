How to execute
=============

| Type  | Parameters | Description | Output |
|:-------------: |:-------------: |:-------------: |:-------------: |
| Execution example  | <table>  <thead>  <tr>  <th>molecule</th>  <th>method</th>  <th>ao_labels</th>  </tr>  </thead>  <tbody>  <tr>  <td>H2</td><td>Qdrift</td><td>C 2p</td>  </tr>  <tr>  </tbody>  </table>     | How to execute one molecule (by name) with one method | T gate cost / Synthesis time  |

Command
-------------
```
python main.py water qdrift 'C 2p'
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

<i> HF energy, MP2 energy, CCSD energy -75.98338639452736 -76.11330094139528 -76.1198661900739
The cost to calculate the energy of water with method qdrift is 1.36e+19 T gates

** -------------------------------------------------- **
**                                                    **
** Execution time     => 0:00:02.685403  in hh:mm:ss  **
********************************************************

```