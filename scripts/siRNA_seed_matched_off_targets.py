import os
import pandas as pd
import subprocess
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from Data_pipeline import TargetPredictionDataset, CharacterTokenizer

from Bio import Align
from Bio.pairwise2 import align as pw2_align, format_alignment

from Global_parameters import PROJ_HOME, RNACOFOLD_BIN, RNAHYBRID_BIN
from DTEA_model import TargetGenerationModel
from Table2Three_scorers import Discriminator_scorer, RNACofold_scorer, RNAHybrid_scorer

def reverse_complement(seq):
    # reverse the sequence and complement the base
    comp = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join(comp.get(b) for b in reversed(seq))

def find_seed_matched_off_targets(target_seed_region, test_mrnas_df, target_clusters, max_samples=2000):
    """
    Find the off-targets that contain the target seed region
    """
    found_off_targets = []
    for i, row in test_mrnas_df.iterrows():
        mrna_seq = row['gene']
        if row['gene_cluster_ID'] not in target_clusters and target_seed_region in mrna_seq:
            found_off_targets.append(mrna_seq)
        if len(found_off_targets) >= max_samples: # stop after finding max_samples off-targets
            break
    return found_off_targets

# ── Paths ──────────────────────────────────────────────────────────────────────
si_rna_targets_path = os.path.join(PROJ_HOME, "siRNA_targets_data", "fda_sirna_targets.csv")
test_mrnas_path = os.path.join(PROJ_HOME, "Manakov2022", "AGO2_eCLIP_Manakov2022_test.tsv.gz")

# ── Evaluate the generated miRNAs ──────────────────────────────────────────────────
print(f"Evaluating the generated miRNAs...")
# load generated mirnas and target mrnas from sirna_targets_path
sirna_targets_df = pd.read_csv(si_rna_targets_path, sep='\t')
test_mrnas_df = pd.read_csv(test_mrnas_path, sep='\t', compression='gzip')
n_complementarity_matched = 0
off_target_dict = {"drug_name": [], "gene_name": [], "noncodingRNA": [], "generated_mirna": [], "off_target_gene": []}
max_samples = 2000
save_path = os.path.join(PROJ_HOME, "siRNA_targets_data", "off_targets.tsv.gz")

for i, row in sirna_targets_df.iterrows():
    drug_name = row["drug"]
    generated_mirna_3to5 = row["generated_mirna"]
    ground_truth_sirna = row["noncodingRNA"]
    generated_mirna_5to3 = generated_mirna_3to5[::-1]
    fda_sirna = row["noncodingRNA"]
    target_mRNA = row["gene"]
    # sample mRNAs targets from Manakov2022 test set that share the same seed regions as the siRNA target mRNAs
    target_seed_region = reverse_complement(fda_sirna[1:8]) # 7-mer seed region
    print(f"Target seed region: ", target_seed_region)
    # Find which gene_cluster_IDs correspond to the target gene
    # by checking which clusters contain the target mRNA sequence
    target_clusters = test_mrnas_df[
        test_mrnas_df["gene"].str.contains(target_mRNA, regex=False, na=False)
    ]["gene_cluster_ID"].unique().tolist() # confirmed no targets in Manakov2022 test set contains sense strand of the siRNA target mRNA
    off_targets = find_seed_matched_off_targets(
        target_seed_region=target_seed_region, 
        test_mrnas_df=test_mrnas_df, 
        target_clusters=target_clusters,
        max_samples=max_samples
    )
    print(f"Found {len(off_targets)} off-targets")
    # save off-targets to a csv file
    for off_target in off_targets:
        off_target_dict["drug_name"].append(drug_name)
        off_target_dict["gene_name"].append(row["gene_name"])
        off_target_dict["noncodingRNA"].append(fda_sirna)
        off_target_dict["generated_mirna"].append(generated_mirna_5to3)
        off_target_dict["off_target_gene"].append(off_target)

off_target_df = pd.DataFrame(off_target_dict)
off_target_df.to_csv(save_path, sep='\t', compression='gzip', index=False)
print(f"Off-targets saved to {save_path}")