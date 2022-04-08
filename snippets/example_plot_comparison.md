How to generate plot
=============

| Type  | Parameters | Description | Output |
|:-------------: |:-------------: |:-------------: |:-------------: |
| Generation plot example  | <table>  <thead>  <tr>  <th>molecule</th>  <th>method</th> </tr>  </thead>  <tbody>  <tr>  <td>All</td><td>All available methods</td></tr>  <tr>  </tbody>  </table>     | How to generate the plot comparison of T gate estimation for all methods | Plot |

Code
-------------
```

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../results/results.csv', delimiter  = ',', index_col = 'Methods')
df.columns = [r'H$_2$', 'HF', r'H$_2$O', r'NH$_3$', r'CH$_4$', r'O$_2$', r'CO$_2$', 'NaCl']

fig, ax = plt.subplots()

plt.ylabel("T-gate cost")
plt.yscale('log')
plt.xticks(ticks = range(len(df.index)), labels = df.index, rotation=90)

for col,color in zip(df.columns, ['gray','brown','m','red','darkorange', 'green','blue', 'black']):
    ax.scatter(df.index, df[col], label = col,facecolors="None", edgecolors=color, linewidth = 1)

ax.legend(bbox_to_anchor=(1, 1))

plt.show()

```

Output
---------

![](https://github.com/PabloAMC/TFermion/blob/develop/snippets/cost_plot.png)
