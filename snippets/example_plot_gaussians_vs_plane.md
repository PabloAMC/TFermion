How to generate plot
=============

| Type  | Parameters | Description | Output |
|:-------------: |:-------------: |:-------------: |:-------------: |
| Generation plot example  | Plane waves vs gaussians  | How to generate the plot comparison between plane waves and gaussians | Plot |

Code
-------------
```

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../results/results.csv', delimiter  = ',', index_col = 'Methods')

fig, (ax2, ax1) = plt.subplots(2, 1, sharex=True,figsize=(7,7))
ax1.scatter(df.T.index, df.T['taylor_naive'], c = 'blue', label = 'Gaussian')
ax1.scatter(df.T.index, df.T['low_depth_taylor'], c = 'red', label = 'Plane waves')
ax1.set_title('Taylor naive algorithm')
ax1.set_yscale('log')
ax1.set_ylabel('T-gate cost')
ax1.legend(bbox_to_anchor=(0.28, 1))

ax2.scatter(df.T.index, df.T['taylor_on_the_fly'], c = 'blue', label = 'Gaussian')
ax2.scatter(df.T.index, df.T['low_depth_taylor_on_the_fly'], c = 'red', label = 'Plane waves')
ax2.set_title('Taylor on-the-fly algorithm')
ax2.legend(bbox_to_anchor=(0.28, 1))
ax2.set_yscale('log')
ax2.set_ylabel('T-gate cost')

plt.show()
```

Output
---------

![](https://github.com/PabloAMC/TFermion/blob/develop/snippets/gaussian_vs_plane.jpg)
