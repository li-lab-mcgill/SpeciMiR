import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from Global_parameters import PROJ_HOME

data_dir = os.path.join(PROJ_HOME, "TargetScan_dataset")
predicted_targets_f = "Mouse_Predicted_Targets_Context_Scores.default_predictions.txt.zip"

# positive miRNA and mRNA pairs
path = os.path.join(data_dir, predicted_targets_f)
predicted_targets = pd.read_csv(path, sep='\t', compression="zip")
# filter for human (9606), chimpanzee (9598), mouse (10090)
tax_ids = [10090]
top_predicted_targets = predicted_targets[
    (predicted_targets["Gene Tax ID"].isin(tax_ids)) &
    (predicted_targets["context++ score percentile"] >= np.int64(80)) # plot top 80% targets
    ]
# filter out non-canonical sites
top_predicted_targets = top_predicted_targets.loc[~top_predicted_targets["Site Type"].isin([-2,-3])]

mouse_data_path = os.path.join(data_dir, "mouse_positive_samples_30_randomized_start.csv")
mouse_train = pd.read_csv(mouse_data_path, sep='\t')
mouse_train = mouse_train.rename(columns={'miRNA ID':'miRNA'})
mouse_train_filtered = (
    mouse_train
    .merge(top_predicted_targets[['Transcript ID','miRNA']].drop_duplicates(),
           on=['Transcript ID','miRNA'],
           how='inner')
)
# mouse_train_sorted = mouse_train_filtered.sort_values(by='seed start', ascending=True).reset_index(drop=True)

# human_data_path = os.path.join(data_dir, "positive_samples_30_randomized_start.csv")
# human_train = pd.read_csv(human_data_path, sep='\t')
# human_train_filtered = human_train.loc[train["Transcript ID"] == Transcript_ID]
# train_sorted = train_filtered.sort_values(by='seed start', ascending=True).reset_index(drop=True)

max_pos = 30 # mRNA segment length

# ——— 2) Build binary matrix ——————————————————————————————————————
# rows = sequences, cols = positions 0…max_pos-1
mat = np.zeros((len(mouse_train_filtered), max_pos), dtype=int)
for i, (s, e) in enumerate(zip(mouse_train_filtered["seed start"], mouse_train_filtered["seed end"])):
    mat[i, s : e+1] = 1

# ——— 3) Plot heatmap ——————————————————————————————————————————
# color non-seed and seed
cmap = ListedColormap(["darkblue", "moccasin"])

fig, ax = plt.subplots(1, 1, figsize=(7, 9))
im = ax.imshow(mat, aspect="auto", cmap=cmap, interpolation="nearest")

# Axis labels & title
ax.set_xlabel("mRNA Sequence Length")
ax.set_ylabel("Predicted Targets")
ax.set_title("Top 80th percentile Seed\nrandomly positioned in 30nt Mouse mRNA")

# Optional colorbar legend
cbar = fig.colorbar(im, ax=ax, ticks=[0, 1])
cbar.set_label("Seed region")
cbar.set_ticklabels(["no", "yes"])

file_name = os.path.join(data_dir, f"80th_percentile_randomized_seed_start.png")
fig.savefig(file_name)
print(f"figure saved to {file_name}")
