#!/usr/bin/env python
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

y_true = []
y_pred = []
with open('/home/chehe/Experiments/groupActivity/Shift-GCN/work_dir/ACMMM/fineturn/Ours_Volleyball_split0.1_classofoer/best_result.txt') as f:
    while 1:
        line = f.readline()
        if not line:
            break
        sample_results = line.strip().split(',')
        y_pred.append(sample_results[0])
        y_true.append(sample_results[1])

matrix = confusion_matrix(y_true, y_pred)
# print(matrix)
matrix = (matrix.T/np.sum(matrix, 1)).T
# print(matrix)
label = ['r_set', 'r_spike', 'r_pass', 'r_winpoint', 'l_set', 'l_spike',  'l_pass', 'l_winpoint']
df = pd.DataFrame(matrix, index=label, columns=label)

plt.figure(figsize=(7.5, 6.3))
ax = sns.heatmap(df, xticklabels=df.corr().columns, yticklabels=df.corr().columns, cmap='GnBu', linewidths=2, annot=True)

plt.xticks(fontsize=12, family='Times New Roman', rotation=45)
plt.yticks(fontsize=12, family='Times New Roman', rotation=45)

plt.tight_layout()
plt.show()
