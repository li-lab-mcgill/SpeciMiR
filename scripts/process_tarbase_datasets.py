#!/usr/bin/env python3
"""
Script to process TarBase datasets and generate merged dataset with 500nt mRNA segments.
Uses get_window_len_mrna function from Target_Scan_dataset_generate.py
"""

import os
import pandas as pd
import sys
import random
from Global_parameters import PROJ_HOME

def get_window_len_mrna(mRNA_seq, window_len, seed_start, seed_end):
    """
    Extract a window-length mRNA segment centered around the seed region.
    This is a standalone version of the function from Target_Scan_dataset_generate.py
    """
    if len(mRNA_seq) < window_len:
        return {"mRNA sequence": mRNA_seq,
                "seed start": seed_start,
                "seed end": seed_end}

    seq_len = len(mRNA_seq)
    seed_len = seed_end - seed_start + 1
    total_ext = window_len - seed_len

    # segment mRNA seq binding site
    seed_start = seed_start - 1  # because index starting from 1
    seed_end = seed_end - 1
    ext_left_max = total_ext
    ext_left_min = 0
    ext_left = random.randint(ext_left_min, ext_left_max)
    ext_left = min(ext_left, seed_start)
    ext_right = total_ext - ext_left
    window_start = seed_start - ext_left
    window_end = seed_end + ext_right + 1  # because window end needs to include the last nucleotide
    
    if window_end > seq_len:
        window_end = seq_len
        window_start = window_end - window_len  # to ensure mRNA seq is `window_len` long
        ext_right = seq_len - seed_end
        ext_left = total_ext - ext_right  # to ensure total extention = ext_right + ext_left
    
    new_seed_start = ext_left
    new_seed_end = new_seed_start + seed_len - 1  # because seed start and seed end needs to be index 
    mRNA_seg = mRNA_seq[window_start:window_end]
    
    if len(mRNA_seg) >= 6:  # must be greater than the shortest seed length
        return {"mRNA sequence": mRNA_seg,
                "seed start": new_seed_start,
                "seed end": new_seed_end}
    else:
        print("Sequence segment is shorter than the shortest seed length 6. Returning None")
        return {"mRNA sequence": None}

def process_tarbase_dataset(file_path, species_name):
    """
    Process a TarBase dataset file and extract 500nt mRNA segments.
    
    Args:
        file_path (str): Path to the TarBase dataset file
        species_name (str): Name of the species (e.g., 'human', 'mouse')
    
    Returns:
        list: List of dictionaries containing processed data
    """
    print(f"Processing {species_name} dataset: {file_path}")
    
    # Read the dataset
    df = pd.read_csv(file_path, sep='\t')
    print(f"Loaded {len(df)} records from {species_name} dataset")
    print(f"Column names: {df.columns}")
    
    processed_data = []
    window_len = 500
    
    for idx, row in df.iterrows():
        if idx % 10000 == 0:
            print(f"Processing {species_name} record {idx}/{len(df)}")
        
        # Extract data from the row
        mirna_id = row['mirna']
        transcript_id = row['ensembl_transcript_id']
        mirna_sequence = row['mirna_sequence']
        mrna_sequence = row['mrna_sequence']
        seed_start = int(row['seed start'])
        seed_end = int(row['seed end'])
        
        # Skip if sequences are too short or contain invalid characters
        if len(mrna_sequence) < 6 or 'N' in mrna_sequence:
            continue
            
        # Use get_window_len_mrna to extract 500nt segment
        try:
            result = get_window_len_mrna(
                mRNA_seq=mrna_sequence,
                window_len=window_len,
                seed_start=seed_start,
                seed_end=seed_end
            )
            
            # Check if we got a valid result
            if result and result.get("mRNA sequence") is not None:
                processed_data.append({
                    'Transcript ID': transcript_id,
                    'miRNA ID': mirna_id,
                    'miRNA sequence': mirna_sequence,
                    'mRNA sequence': result['mRNA sequence'],
                    'seed start': result['seed start'],
                    'seed end': result['seed end'],
                    'label': 1  # Positive samples
                })
            else:
                print(f"Skipping {transcript_id}:{mirna_id} - invalid segment")
                break
                
        except Exception as e:
            print(f"Error processing {transcript_id}: {e}")
            continue
    
    print(f"Successfully processed {len(processed_data)} records from {species_name} dataset")
    return processed_data

def main():
    """Main function to process both datasets and create merged output."""
    
    # Set up paths
    tarbase_dir = os.path.join(PROJ_HOME, "TarBase_dataset")
    
    human_file = os.path.join(tarbase_dir, "sampled_canonical_human_microT.tsv")
    mouse_file = os.path.join(tarbase_dir, "sampled_canonical_mouse_microT.tsv")
    output_file = os.path.join(tarbase_dir, "Positive_primates_500_randomized_start.csv")
    
    # Check if input files exist
    if not os.path.exists(human_file):
        print(f"Error: Human dataset file not found: {human_file}")
        return
    
    if not os.path.exists(mouse_file):
        print(f"Error: Mouse dataset file not found: {mouse_file}")
        return
    
    # Process both datasets
    print("Starting processing of TarBase datasets...")
    
    human_data = process_tarbase_dataset(human_file, "human")
    mouse_data = process_tarbase_dataset(mouse_file, "mouse")
    
    # Combine the datasets
    all_data = human_data + mouse_data
    print(f"Total records after processing: {len(all_data)}")
    
    # Convert to DataFrame and save
    if all_data:
        df_merged = pd.DataFrame(all_data)
        
        # Shuffle the data to randomize the order
        df_merged = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save to CSV
        df_merged.to_csv(output_file, index=False)
        print(f"Saved merged dataset to: {output_file}")
        print(f"Final dataset contains {len(df_merged)} records")
        
        # Display summary statistics
        print("\nDataset summary:")
        print(f"Human records: {len(human_data)}")
        print(f"Mouse records: {len(mouse_data)}")
        print(f"Total records: {len(df_merged)}")
        print(f"Unique transcripts: {df_merged['Transcript ID'].nunique()}")
        print(f"Unique miRNAs: {df_merged['miRNA ID'].nunique()}")
        
        # Show first few rows
        print("\nFirst 5 rows of the merged dataset:")
        print(df_merged.head())
        
    else:
        print("No data to save!")

if __name__ == "__main__":
    main()
