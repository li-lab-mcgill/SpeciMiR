#!/usr/bin/env python3
"""
Script to split TarBase dataset using the same ratio as TargetScan dataset
and merge the training, validation, and testing datasets.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    """Main function to split TarBase dataset and merge with TargetScan datasets."""
    
    # Set up paths
    base_dir = "path/to/proj_home"
    tarbase_dir = os.path.join(base_dir, "TarBase_dataset")
    targetscan_dir = os.path.join(base_dir, "TargetScan_dataset")
    
    # Input files
    tarbase_file = os.path.join(tarbase_dir, "Positive_primates_500_randomized_start.csv")
    targetscan_train_file = os.path.join(targetscan_dir, "Positive_primates_train_500_randomized_start.csv")
    targetscan_val_file = os.path.join(targetscan_dir, "Positive_primates_validation_500_randomized_start.csv")
    targetscan_test_file = os.path.join(targetscan_dir, "Positive_primates_test_500_randomized_start.csv")
    
    # Output files
    merged_train_file = os.path.join(targetscan_dir, "Merged_primates_train_500_randomized_start.csv")
    merged_val_file = os.path.join(targetscan_dir, "Merged_primates_validation_500_randomized_start.csv")
    merged_test_file = os.path.join(targetscan_dir, "Merged_primates_test_500_randomized_start.csv")
    
    print("Loading datasets...")
    
    # Load TarBase dataset
    print(f"Loading TarBase dataset: {tarbase_file}")
    tarbase_df = pd.read_csv(tarbase_file)
    print(f"TarBase dataset: {len(tarbase_df)} records")
    
    # Load TargetScan datasets to get the split ratios
    print(f"Loading TargetScan datasets...")
    targetscan_train = pd.read_csv(targetscan_train_file)
    targetscan_val = pd.read_csv(targetscan_val_file)
    targetscan_test = pd.read_csv(targetscan_test_file)
    
    targetscan_total = len(targetscan_train) + len(targetscan_val) + len(targetscan_test)
    train_ratio = len(targetscan_train) / targetscan_total
    val_ratio = len(targetscan_val) / targetscan_total
    test_ratio = len(targetscan_test) / targetscan_total
    
    print(f"TargetScan dataset split ratios:")
    print(f"  Training: {train_ratio:.3f} ({len(targetscan_train)} records)")
    print(f"  Validation: {val_ratio:.3f} ({len(targetscan_val)} records)")
    print(f"  Testing: {test_ratio:.3f} ({len(targetscan_test)} records)")
    
    # Split TarBase dataset using the same ratios
    print(f"\nSplitting TarBase dataset using the same ratios...")
    
    # First split: separate training from validation+test
    tarbase_train, tarbase_rem = train_test_split(
        tarbase_df, 
        test_size=(val_ratio + test_ratio), 
        random_state=42, 
        shuffle=True
    )
    
    # Second split: separate validation from test
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    tarbase_val, tarbase_test = train_test_split(
        tarbase_rem, 
        test_size=(1 - val_test_ratio), 
        random_state=42, 
        shuffle=False
    )
    
    print(f"TarBase dataset split:")
    print(f"  Training: {len(tarbase_train)} records")
    print(f"  Validation: {len(tarbase_val)} records")
    print(f"  Testing: {len(tarbase_test)} records")
    
    # Merge datasets
    print(f"\nMerging datasets...")
    
    # Merge training datasets
    merged_train = pd.concat([targetscan_train, tarbase_train], ignore_index=True)
    print(f"Merged training dataset: {len(merged_train)} records")
    
    # Merge validation datasets
    merged_val = pd.concat([targetscan_val, tarbase_val], ignore_index=True)
    print(f"Merged validation dataset: {len(merged_val)} records")
    
    # Merge testing datasets
    merged_test = pd.concat([targetscan_test, tarbase_test], ignore_index=True)
    print(f"Merged testing dataset: {len(merged_test)} records")
    
    # Shuffle the merged datasets
    print(f"\nShuffling merged datasets...")
    merged_train = merged_train.sample(frac=1, random_state=42).reset_index(drop=True)
    merged_val = merged_val.sample(frac=1, random_state=42).reset_index(drop=True)
    merged_test = merged_test.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save merged datasets
    print(f"\nSaving merged datasets...")
    merged_train.to_csv(merged_train_file, index=False)
    merged_val.to_csv(merged_val_file, index=False)
    merged_test.to_csv(merged_test_file, index=False)
    
    print(f"Saved merged datasets:")
    print(f"  Training: {merged_train_file}")
    print(f"  Validation: {merged_val_file}")
    print(f"  Testing: {merged_test_file}")
    
    # Display summary statistics
    print(f"\nFinal dataset summary:")
    print(f"TargetScan + TarBase merged datasets:")
    print(f"  Training: {len(merged_train)} records")
    print(f"  Validation: {len(merged_val)} records")
    print(f"  Testing: {len(merged_test)} records")
    print(f"  Total: {len(merged_train) + len(merged_val) + len(merged_test)} records")
    
    # Check for unique transcripts and miRNAs
    print(f"\nUnique counts in merged training dataset:")
    print(f"  Unique transcripts: {merged_train['Transcript ID'].nunique()}")
    print(f"  Unique miRNAs: {merged_train['miRNA ID'].nunique()}")
    
    # Show first few rows of merged training dataset
    print(f"\nFirst 3 rows of merged training dataset:")
    print(merged_train.head(3))

if __name__ == "__main__":
    main()
