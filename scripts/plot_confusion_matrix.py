import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Global_parameters import PROJ_HOME

# 1. Read csv
df = pd.read_csv(os.path.join(PROJ_HOME, "Performance/TargetScan_test/TwoTowerTransformer/30/binding_span_predictions.csv"))

df = df.loc[df["label"] == 1]
print(len(df))

# 2. calculate true and predicted seed length
df['true_len'] = df['seed end'] - df['seed start'] + 1
df['pred_len'] = df['pred end'] - df['pred start'] + 1

# 3. only keep 6-mer、7-mer、8-mer
# valid = [6, 7, 8]
# df = df[df['true_len'].isin(valid) & df['pred_len'].isin(valid)]

# 4. construct confusion matrix
pred_labels = df["pred_len"].unique()
true_labels = df["true_len"].unique()
cm = pd.crosstab(
    df['true_len'], 
    df['pred_len'], 
    rownames=['True'], 
    colnames=['Pred'], 
    dropna=False
)
cm = cm.reindex(index=true_labels, columns=pred_labels)

# 5. plot and save heatmap
fig, ax = plt.subplots(figsize=(25,3))
im = ax.imshow(cm.values, cmap="Blues", interpolation='nearest', aspect='auto')

ax.set_xticks(range(len(pred_labels)))
ax.set_xticklabels([f'{l}-mer' for l in pred_labels], rotation=30, ha='right')
ax.set_yticks(range(len(true_labels)))
ax.set_yticklabels([f'{l}-mer' for l in true_labels])
ax.set_xlabel('Predicted Seed Length')
ax.set_ylabel('True Seed Length')

# label cells with numbers
thresh = cm.values.max() / 2
for i in range(len(true_labels)):
    for j in range(len(pred_labels)):
        count = cm.values[i, j]
        color = 'white' if count > thresh else 'black'
        plt.text(j, i, count, ha='center', va='center', color=color)


# pad=0.02, colorbar 
cbar = fig.colorbar(im, ax=ax, pad=0.01, label='Count')
plt.tight_layout()

# save plots
performance_dir = os.path.join(PROJ_HOME, "Performance/TargetScan_test/TwoTowerTransformer/30/")
plt.savefig(os.path.join(performance_dir, 'confusion_matrix_w_cnn.png'), dpi=500, bbox_inches='tight')
print(f"plot saved to {performance_dir}")
