import os
import pandas as pd
import numpy as np
import random
from Global_parameters import PROJ_HOME

# Define paths
datadir = "miR_degradome_ago_clip_pairing_data"
pairs = os.path.join(datadir, "paired_output_500_sequence.csv")
out = os.path.join(datadir, "paired_500_randomized_start.csv")
targetscan_negatives = os.path.join(PROJ_HOME, "TargetScan_dataset/TargetScan_train_500_randomized_start.csv")

# Random seed for reproducibility
random_seed = 42

def main():
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    print(f"[INFO] Loading positive pairs from {pairs}")
    positive_df = pd.read_csv(pairs)
    
    print(f"[INFO] Loaded {len(positive_df)} positive pairs")
    print(f"[INFO] Columns: {positive_df.columns.tolist()}")
    
    # Add label column to positive pairs
    positive_df['label'] = 1
    positive_df.columns = ["Transcript ID", "miRNA ID", "mRNA sequence", "miRNA sequence","seed start", "seed end", "cleave site", "label"]
    
    print(f"\n[INFO] Loading TargetScan dataset from {targetscan_negatives}")
    targetscan_df = pd.read_csv(targetscan_negatives)
    
    print(f"[INFO] Loaded {len(targetscan_df)} TargetScan samples")
    print(f"[INFO] TargetScan columns: {targetscan_df.columns.tolist()}")
    
    # Filter for negative samples (label=0)
    if 'label' in targetscan_df.columns:
        negative_candidates = targetscan_df[targetscan_df['label'] == 0].copy()
        print(f"[INFO] Found {len(negative_candidates)} negative samples in TargetScan dataset")
    else:
        print("[ERROR] 'label' column not found in TargetScan dataset")
        print(f"[INFO] Available columns: {targetscan_df.columns.tolist()}")
        raise ValueError("TargetScan dataset must have a 'label' column")
    
    # Check if we have enough negative samples
    num_negatives_needed = len(positive_df)
    if len(negative_candidates) < num_negatives_needed:
        print(f"[WARNING] Not enough negative samples. Need {num_negatives_needed}, have {len(negative_candidates)}")
        print(f"[WARNING] Will use all available negative samples")
        negative_df = negative_candidates
    else:
        # Randomly sample equal number of negative samples
        print(f"[INFO] Randomly sampling {num_negatives_needed} negative samples from {len(negative_candidates)} candidates")
        negative_df = negative_candidates.sample(n=num_negatives_needed, random_state=random_seed).copy()
    
    print(f"[INFO] Selected {len(negative_df)} negative samples")

    # add seed start, seed end, cleave site columns to negative df
    negative_df['seed start'] = -1
    negative_df['seed end'] = -1
    negative_df['cleave site'] = -1
    
    # Ensure both dataframes have the same columns
    # Get common columns
    common_cols = set(positive_df.columns) & set(negative_df.columns)
    print(f"\n[INFO] Common columns: {common_cols}")
    
    # Check required columns
    required_cols = ['Transcript ID', 'miRNA ID', 'mRNA sequence', 'miRNA sequence', 'seed start', 'seed end', 'cleave site','label']
    for col in required_cols:
        if col not in common_cols:
            print(f"[WARNING] Column '{col}' not found in both datasets")
    
    try:
        # Keep only common columns for alignment
        positive_df = positive_df[required_cols]
        negative_df = negative_df[required_cols]
    except Exception as e:
        print(f"[ERROR] Failed to align columns: {e}")
        raise e
    
    # Combine positive and negative pairs
    print(f"\n[INFO] Combining positive and negative pairs...")
    combined_df = pd.concat([positive_df, negative_df], ignore_index=True)
    
    # Shuffle the combined dataset
    print(f"[INFO] Shuffling combined dataset...")
    combined_df = combined_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Write output
    combined_df.to_csv(out, index=False)
    
    print(f"\n[SUMMARY]")
    print(f"  Positive pairs: {len(positive_df)}")
    print(f"  Negative pairs: {len(negative_df)}")
    print(f"  Total pairs: {len(combined_df)}")
    print(f"  Positive label count: {(combined_df['label'] == 1).sum()}")
    print(f"  Negative label count: {(combined_df['label'] == 0).sum()}")
    print(f"  Output written to: {out}")
    print(f"[OK] Done!")

if __name__ == "__main__":
    main()
