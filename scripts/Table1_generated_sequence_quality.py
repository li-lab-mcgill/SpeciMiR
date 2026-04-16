'''
This script will generate miRNA sequences given mRNA sequences and evaluate the quality of the generated miRNA sequences by:
1. Token-level accuracy
2. Sequence diversity: % of unique sequences generated / total generated sequences
3. Seed match rate: % of canonical seed matches in the generated miRNA sequences
'''

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from DTEA_model import TargetGenerationModel
from Data_pipeline import TargetPredictionDataset, CharacterTokenizer
from Global_parameters import PROJ_HOME
from seed_match_verification import verify_anywhere_seed_matches

def eval_token_accuracy(dataset):
    total_correct_tokens = 0
    total_tokens = 0
    for index, row in dataset.iterrows():
        true_mirna = list(row["noncodingRNA"])
        generated_mirna = list(row["generated_mirna"])
        min_length = min(len(true_mirna), len(generated_mirna))
        total_correct_tokens += sum([1 for i in range(min_length) if true_mirna[i] == generated_mirna[i]])
        total_tokens += len(true_mirna)
    return total_correct_tokens / total_tokens

def eval_sequence_diversity(dataset, mirna_col="generated_mirna"):
    total_sequences = len(dataset)
    unique_sequences = len(set(dataset[mirna_col]))
    print(f"unique sequences: {unique_sequences}, total sequences: {total_sequences}")
    return unique_sequences / total_sequences

def eval_seed_match_rate(dataset, mirna_col="generated_mirna"):
    """
    Evaluate the fraction of sequences that have at least one canonical seed match
    Save the found seed match types as an additional column in the dataset    
    """
    total_sequences = len(dataset)
    valid_6mers = 0
    valid_7mers_a1 = 0
    valid_7mers_m8 = 0
    valid_8mers = 0
    found_seed_types = []
    for index, row in dataset.iterrows():
        mrna_seq = row["gene"]
        mirna_seq = row[mirna_col]
        found, m_type, _, _ = verify_anywhere_seed_matches(mrna_seq, mirna_seq) # can switch to verify_seed_matches for exact seed matches at 2-8nt
        if found:
            if m_type == "6-mer":
                valid_6mers += 1
            elif m_type == "7-mer-A1":
                valid_7mers_a1 += 1
            elif m_type == "7-mer-m8":
                valid_7mers_m8 += 1
            elif m_type == "8-mer":
                valid_8mers += 1
            found_seed_types.append(m_type)
        else:
            found_seed_types.append("NA")
    # percentage of valid 6-mers, 7-mers-A1, 7-mers-m8, and 8-mers
    total_matches = valid_6mers + valid_7mers_a1 + valid_7mers_m8 + valid_8mers
    print(f"Total matches: {total_matches / total_sequences:.2%}") # percentage of total matches
    print(f"Percentage of valid 6-mers in [{mirna_col}]: {valid_6mers / total_sequences:.2%}") # percentage of valid 6-mers
    print(f"Percentage of valid 7-mers-A1 in [{mirna_col}]: {valid_7mers_a1 / total_sequences:.2%}") # percentage of valid 7-mers-A1
    print(f"Percentage of valid 7-mers-m8 in [{mirna_col}]: {valid_7mers_m8 / total_sequences:.2%}") # percentage of valid 7-mers-m8
    print(f"Percentage of valid 8-mers in [{mirna_col}]: {valid_8mers / total_sequences:.2%}") # percentage of valid 8-mers
    dataset["seed_types"] = found_seed_types
    
    return total_matches / total_sequences, dataset # percentage of total matches

def main():
    # ─── Configuration ────────────────────────────────────────────────────────────
    MRNA_MAX_LEN = 80
    MIRNA_MAX_LEN = 26 # 24 + 2
    BATCH_SIZE = 64
    DEVICE = "cuda:4"
    SEED = 42
    EMBED_DIM = 1024
    NUM_HEADS = 8
    NUM_LAYERS = 4
    FF_DIM = 4096
    USE_LONGFORMER = True
    # save_path = None
    save_path = os.path.join(PROJ_HOME, "Manakov2022/AGO2_eCLIP_Manakov2022_test_predicted.csv.gz")

    # datapath = os.path.join(PROJ_HOME, "Manakov2022/AGO2_eCLIP_Manakov2022_test.tsv.gz")
    # ckpt_path = os.path.join(PROJ_HOME, "checkpoints/specificity_gen/Manakov2022_train/best_loss_0.5440_epoch6.pth")
    
    # dataset = pd.read_csv(datapath, compression="gzip", sep="\t")
    # # filter for only target mRNAs
    # dataset = dataset[dataset["label"] == 1]
    # tokenizer = CharacterTokenizer(characters=["A", "T", "C", "G", "N"],
    #                 add_special_tokens=False, 
    #                 model_max_length=MRNA_MAX_LEN-2, # minus 2 for BOS and EOS tokens
    #                 padding_side="right")
    # ds_test = TargetPredictionDataset(data=dataset,
    #                     mrna_max_len=MRNA_MAX_LEN,
    #                     mirna_max_len=MIRNA_MAX_LEN,
    #                     tokenizer=tokenizer,
    #                     mRNA_col="gene",
    #                     miRNA_col="noncodingRNA")
    # test_dataloader = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

    # model = TargetGenerationModel(
    #     mirna_max_len=MIRNA_MAX_LEN,
    #     mrna_max_len=MRNA_MAX_LEN,
    #     embed_dim=EMBED_DIM,
    #     num_heads=NUM_HEADS,
    #     num_layers=NUM_LAYERS,
    #     ff_dim=FF_DIM,
    #     use_longformer=USE_LONGFORMER
    # )
    # sd = torch.load(ckpt_path, map_location=DEVICE)
    # model.load_state_dict(sd["generator_state_dict"])
    # print(f"Model loaded from {ckpt_path}")
    # model.device = DEVICE
    # model.to(DEVICE)
    # model.eval()

    # print(f"Generating miRNA sequences for {len(dataset)} target mRNAs...")
    # n_done = 0
    # all_generated_seqs = []
    # for batch in test_dataloader:
    #     mrna_tokens = batch["mrna_input_ids"].to(DEVICE)
    #     generated = model.greedy_generate(model=model, 
    #                                     device=DEVICE,
    #                                     mrna_tokens=mrna_tokens, 
    #                                     max_len=MIRNA_MAX_LEN)
    #     # Decode
    #     for i in range(generated.size(0)):
    #         seq_ids = generated[i].tolist()
    #         decoded = tokenizer.decode(seq_ids, skip_special_tokens=True)
    #         all_generated_seqs.append(decoded)
    #     n_done += generated.size(0)
    #     print(f"  Generated {n_done}/{len(dataset)} sequences", end="\r", flush=True)
    # print()
    # dataset["generated_mirna"] = all_generated_seqs
    # if save_path is not None:
    #     dataset.to_csv(save_path, compression="gzip", index=False, sep='\t')
    #     print(f"Generated mirna saved to {save_path}")

    # --------Evaluate the quality of the generated miRNA sequences --------
    dataset_path = os.path.join(PROJ_HOME, "Manakov2022/AGO2_eCLIP_Manakov2022_test_predicted.csv.gz")
    dataset = pd.read_csv(dataset_path, compression="gzip", sep="\t")
    # 1. Token-level accuracy
    # compare the generated miRNA sequences with the true miRNA sequences token-by-token
    # avg_token_accuracy = eval_token_accuracy(dataset)

    # 2. Sequence diversity
    # % of unique sequences generated / total generated sequences
    seq_diversity = eval_sequence_diversity(dataset, mirna_col="generated_mirna")
    seq_diversity_gt = eval_sequence_diversity(dataset, mirna_col="noncodingRNA")

    # 3. Seed match rate
    # % of canonical seed matches in the generated miRNA sequences
    seed_match_rate, dataset = eval_seed_match_rate(dataset, mirna_col="generated_mirna") # seed types are saved in the dataset
    seed_match_rate_gt, _ = eval_seed_match_rate(dataset, mirna_col="noncodingRNA")

    # print(f"Sequence identity vs ground truth: {avg_token_accuracy:.2%}")
    print("Mean length:", dataset["generated_mirna"].str.len().mean())
    print("Ground truth mean length:", dataset["noncodingRNA"].str.len().mean())
    print(f"Sequence diversity: {seq_diversity:.2%}")
    print(f"Sequence diversity (ground truth): {seq_diversity_gt:.2%}")
    print(f"Seed match rate: {seed_match_rate:.2%}")
    print(f"Seed match rate (ground truth): {seed_match_rate_gt:.2%}")
    
    if save_path is not None:
        dataset.to_csv(save_path, compression="gzip", index=False, sep='\t')
        print(f"Generated mirna saved to {save_path}")

if __name__ == "__main__":
    main()