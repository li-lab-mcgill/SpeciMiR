import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Global_parameters import PROJ_HOME

# 1. 读取 CSV（替换为你的文件路径）
df = pd.read_csv(os.path.join(PROJ_HOME, "Performance/TargetScan_test/TwoTowerTransformer/30/binding_span_predictions.csv"))

df = df.loc[df["label"] == 1]
print(len(df))

# 2. 计算真实和预测的 seed length
df['true_len'] = df['seed end'] - df['seed start'] + 1
df['pred_len'] = df['pred end'] - df['pred start'] + 1

# 3. 只保留 6-mer、7-mer、8-mer
# valid = [6, 7, 8]
# df = df[df['true_len'].isin(valid) & df['pred_len'].isin(valid)]

# 4. 构建混淆矩阵
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

# 5. 绘制并保存热图
fig, ax = plt.subplots(figsize=(25,3))
im = ax.imshow(cm.values, cmap="Blues", interpolation='nearest', aspect='auto')

# 调整 x/y 轴等...
ax.set_xticks(range(len(pred_labels)))
ax.set_xticklabels([f'{l}-mer' for l in pred_labels], rotation=30, ha='right')
ax.set_yticks(range(len(true_labels)))
ax.set_yticklabels([f'{l}-mer' for l in true_labels])
ax.set_xlabel('Predicted Seed Length')
ax.set_ylabel('True Seed Length')

# 在方格中添加数值标注
thresh = cm.values.max() / 2
for i in range(len(true_labels)):
    for j in range(len(pred_labels)):
        count = cm.values[i, j]
        color = 'white' if count > thresh else 'black'
        plt.text(j, i, count, ha='center', va='center', color=color)


# 这里 pad=0.02，把 colorbar 往 heatmap 更靠近
cbar = fig.colorbar(im, ax=ax, pad=0.01, label='Count')
plt.tight_layout()

# 保存
performance_dir = os.path.join(PROJ_HOME, "Performance/TargetScan_test/TwoTowerTransformer/30/")
plt.savefig(os.path.join(performance_dir, 'confusion_matrix_w_cnn.png'), dpi=500, bbox_inches='tight')
print(f"plot saved to {performance_dir}")
