import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
     (predicted_targets["context++ score percentile"] >= np.int64(80)) # top 20% likely pairs
     ]
# filter out non-canonical sites
top_predicted_targets = top_predicted_targets.loc[~top_predicted_targets["Site Type"].isin([-2,-3])]
# filter for miRNA in training data
data_dir = os.path.join(PROJ_HOME, "TarBase_dataset")
data_path = os.path.join(data_dir, "microT_human_geq_0.85.tsv")
# mouse_data_path = os.path.join(data_dir, "mouse_positive_samples_30_randomized_start.csv")
train = pd.read_csv(data_path, sep='\t')
# train = train.rename(columns={'miRNA ID':'miRNA'})
# mouse_data = pd.read_csv(mouse_data_path)
# train = pd.concat((human_data, mouse_data), axis=0)
# 2) Filter to only those (Transcript ID, miRNA ID) pairs you trained on
# filtered = (
#     top_predicted_targets
#     .merge(train[['Transcript ID','miRNA']].drop_duplicates(),
#            on=['Transcript ID','miRNA'],
#            how='inner')
# )

# 3) Compute, for each miRNA, the proportion of each Site Type
#    (group-by miRNA ID then Site Type, normalize by total per miRNA)
prop = (
    train
    .groupby(['mirna','mre_type'])
    .size()
    .groupby(level=0)
    .apply(lambda x: x / x.sum())
    .unstack(fill_value=0)
)

# prop = prop.rename(columns={
#     1: '7mer-1a',
#     2: '7mer-m8',
#     3: '8mer'
# })

# 2) Sort miRNAs by descending 8mer proportion
prop_sorted = prop.sort_values(by='8mer', ascending=False)

# 4) Plot as a stacked bar chart
fig, ax = plt.subplots(figsize=(25,10))
# set only the 6mer color; other columns use default cycle
colors = {"6mer": "tab:cyan", 
        "7mer": "tab:orange",
        "7mer-1a": "tab:blue",
        "7mer-m8": "tab:orange",
        "8mer": "tab:green", 
        "9mer": "tab:brown", 
        "8mer+mismatch": "tab:red",
        "8mer+wobble": "tab:purple"}
prop_sorted.plot(kind='bar', stacked=True, ax=ax, color=colors)

ax.set_xticks([])
ax.set_xlabel("miRNA")
ax.set_ylabel("Proportion of Sites")
ax.set_title("Site-Type Composition per miRNA")
ax.legend(title="Site Type Proportion for Individual miRNA", bbox_to_anchor=(1.02,1), loc="upper left")

fig.savefig(os.path.join(data_dir, "human_microT_Site_Types_proportion.png"), dpi=500, bbox_inches='tight')

