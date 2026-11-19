# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 19:16:13 2026

@author: umroot
"""

import numpy as np
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt

mae_A = np.array([18.497885763645172, 

21.62558287382126, 

19.981257617473602, 

19.1830113530159, 

21.6160386800766, 

22.700104117393494, 

19.886142015457153, 

24.289408326148987, 

19.29713487625122, 

18.93012225627899])

mae_B = np.array([19.88876163959503, 

24.523499608039856, 

19.27521824836731, 

21.03261947631836, 

24.22790825366974, 

20.564571022987366, 

25.884845852851868, 

22.64757603406906, 

23.094283044338226, 

23.78573566675186])

# Positive values indicate A performs better
diff = mae_B - mae_A

# Paired one-sided t-test
stat, p_value = ttest_rel(mae_B, mae_A, alternative='greater')

print("Mean MAE A:", mae_A.mean())
print("Mean MAE B:", mae_B.mean())
print("Mean improvement:", diff.mean())
print("p-value:", p_value)

if p_value < 0.05:
    print("Model A is significantly better.")
else:
    print("No statistically significant improvement.")
    
    
    
    
#####boxplot
# Example: replace with your actual lists
scores_without = [0.8571, 0.3036, 0.6107, 0.8571, 0.8036, 0.8036, 0.4607, 0.8036, 0.4607, 0.5286, 0.2964]
scores_with = [0.9214, 0.20, 0.6679, 0.9214, 0.8321, 0.8321, 0.5179, 0.8321, 0.5179, 0.7000, 0.8464]

data = [scores_without, scores_with]

plt.figure(figsize=(6, 5))

plt.boxplot(
    data,
    labels=['Without similarity loss', 'With similarity loss'],
    patch_artist=True,
    widths=0.5,
    medianprops=dict(color='black', linewidth=1.5),
    boxprops=dict(facecolor='lightblue', alpha=0.7),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
    flierprops=dict(marker='o', markersize=5)
)

plt.ylabel('Spearman correlation', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()