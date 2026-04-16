"""
This script scores the generated miRNA sequences using three scorers:
1. Manakov-trained Discriminator
2. RNACofold
3. miRBind

for the following pairs:
    1. Real miRNA <-> target mRNA (ceiling)
    2. generated miRNA <-> target mRNA (expected to be high)
    3. generated miRNA <-> off-target mRNA (expected to be low)
    4. Random miRNA <-> target mRNA (reference)
"""
import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from DTEA_model import CrossAttentionPredictor
from Data_pipeline import TargetPredictionDataset, CharacterTokenizer
from Global_parameters import PROJ_HOME
from seed_match_verification import verify_anywhere_seed_matches
from Finetune_discriminator import load_discriminator_checkpoint, FrozenDiscriminator

# ─── Configuration ────────────────────────────────────────────────────────────
MRNA_MAX_LEN = 80
MIRNA_MAX_LEN = 22 
BATCH_SIZE = 64
DEVICE = "cuda:1"
SEED = 42
EMBED_DIM = 1024
NUM_HEADS = 8
NUM_LAYERS = 4
FF_DIM = 4096
USE_LONGFORMER = True
K_OFF = 5
BASES = ["A", "T", "C", "G"]

# Discriminator architecture (must match finetuned checkpoint)
DISC_EMBED_DIM     = 256
DISC_NUM_HEADS     = 2
DISC_NUM_LAYERS    = 2
DISC_FF_DIM        = 512
DISC_VOCAB_SIZE    = 12
DISC_CKPT = os.path.join(PROJ_HOME, 
    "checkpoints/TargetScan/TwoTowerTransformer/CNN-tokenized/Manakov2022_train/50/",
    "best_binding_aps_0.8357_epoch14.pth"
    )

def make_random_mirna(target, mirna_len=22):
    """
    Create a random miRNA that does not contain any seed matches to the target mRNA.
    """
    match_seed = False
    while not match_seed:
        rand_mirna = "".join(random.choices(BASES, k=mirna_len))
        found, _, _, _ = verify_anywhere_seed_matches(target, rand_mirna)
        if not found:
            match_seed = True
    return rand_mirna

def main():
    # ─── Load data ────────────────────────────────────────────────────────────────
    datapath = os.path.join(PROJ_HOME, "Manakov2022/AGO2_eCLIP_Manakov2022_test_predicted.tsv.gz")
    dataset = pd.read_csv(datapath, compression="gzip", sep="\t")
    all_mrnas = dataset["gene"].unique().tolist()
    # filter for only target mRNAs
    dataset = dataset[dataset["label"] == 1]
    # filter for only generated miRNA sequences
    dataset = dataset[dataset["generated_mirna"] != "NA"]
    # filter for only real miRNA sequences
    dataset = dataset[dataset["noncodingRNA"] != "NA"]
    # filter for `K_OFF` off-target mRNAs for each generated miRNA
    gen_off_mirna = []
    gen_off_mrna = []
    for _, row in dataset.iterrows():
        target = row["mrna_seq"]
        gen_seq = row["generated_mirna"]
        # Sample k_off off-targets (excluding the target itself)
        candidates = [m for m in random.sample(all_mrnas, min(K_OFF * 2, len(all_mrnas)))
                       if m != target][:K_OFF]
        while len(candidates) < K_OFF:
            candidates.append(random.choice(all_mrnas))
        for off_mrna in candidates:
            gen_off_mirna.append(gen_seq)
            gen_off_mrna.append(off_mrna)

    # create random miRNA (not seed-matched) for each target mRNA
    random_mirnas = []
    target_mrna = all_mrnas
    for index, row in dataset.iterrows():
        target = row["gene"]    
        rand_mirna = make_random_mirna(target)
        random_mirnas.append(rand_mirna)
    
    # load discriminator model
    disc_model = CrossAttentionPredictor(
        mirna_max_len=MIRNA_MAX_LEN, mrna_max_len=MRNA_MAX_LEN,
        vocab_size=DISC_VOCAB_SIZE, num_layers=DISC_NUM_LAYERS,
        embed_dim=DISC_EMBED_DIM, num_heads=DISC_NUM_HEADS,
        ff_dim=DISC_FF_DIM, hidden_sizes=[DISC_FF_DIM, DISC_FF_DIM],
        n_classes=1, dropout_rate=0.1, device=DEVICE,
        predict_span=False, predict_binding=True, predict_cleavage=False,
        use_longformer=False,
    )
    disc_model = load_discriminator_checkpoint(disc_model, DISC_CKPT, DEVICE)
    discriminator = FrozenDiscriminator(disc_model)



    