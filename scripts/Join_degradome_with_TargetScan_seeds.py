import os
import pandas as pd
from Global_parameters import PROJ_HOME

targetscan_data_dir = os.path.join(PROJ_HOME, "TargetScan_dataset")
degradome_data_dir = os.path.join(PROJ_HOME, "miR_degradome_ago_clip_pairing_data")
predicted_targets_f = "Human_Predicted_Targets_Context_Scores.default_predictions.txt.zip"

# positive miRNA and mRNA pairs
path = os.path.join(targetscan_data_dir, predicted_targets_f)
predicted_targets = pd.read_csv(path, sep='\t', compression="zip")
# filter for human (9606), chimpanzee (9598), mouse (10090)
tax_ids = [9606]
top_predicted_targets = predicted_targets[
    predicted_targets["Gene Tax ID"].isin(tax_ids) 
    ]
# filter out non-canonical sites
top_predicted_targets = top_predicted_targets.loc[~top_predicted_targets["Site Type"].isin([-2,-3])]

positive_pairs = top_predicted_targets[[
     "miRNA",
     "Transcript ID",
     "UTR_start",
     "UTR_end"
]]
positive_pairs.columns = ["miRNA", "Transcript_ID", "UTR_start", "UTR_end"]
positive_pairs = positive_pairs.drop_duplicates(subset=["Transcript_ID", "miRNA", "UTR_start", "UTR_end"])

# read degradome data
degradome_data_path = os.path.join(degradome_data_dir, "starBase_degradome_windows_500.tsv")
degradome_data = pd.read_csv(degradome_data_path, sep='\t')
degradome_data.columns = ["miRNA", "Transcript_ID", "cleave_site_tx0", "mRNA sequence", "miRNA sequence"]

# join targetscan data and degradome data by miRNA and transcript ID
merged_data = pd.merge(degradome_data, positive_pairs, on=["miRNA", "Transcript_ID"], how="inner")

merged_data.to_csv(os.path.join(degradome_data_dir, "degradome_with_targetscan_seeds.tsv"), sep='\t', index=False)


