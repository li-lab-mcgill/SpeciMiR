"""
This script explores the ablation study of generator model.
There are 4 variants of the generator model:
1. Full mode trained with generation loss and specifity loss and off-target loss
2. Generator model trained without generation loss (alpha = 0) 
3. Generator model trained without specificity loss (beta = 0)
4. Generator model trained without off-target loss (gamma = 0)

We will train these 4 variants of the generator model and evaluate the performace on Manakov2022 test set.
"""

from Finetune_Specificity import (
    load_discriminator_checkpoint,
    TargetGenerationModel, 
    SpecificityTrainer,
    SpecificityDataset,
    FrozenDiscriminator,
    Generator,
)

import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from Global_parameters import PROJ_HOME
from DTEA_model import CrossAttentionPredictor

def main():
    # ── Hyperparameters ───────────────────────────────────────────────────
    # Generator architecture (must match pretrained checkpoint)
    MRNA_MAX_LEN  = 80       # must be divisible by 2*window_size=40 for Longformer
    MIRNA_MAX_LEN = 24 + 2    # generator's pretrained miRNA length
    GEN_EMBED_DIM     = 1024
    GEN_NUM_HEADS     = 8
    GEN_NUM_LAYERS    = 4
    GEN_FF_DIM        = 4096
    GEN_VOCAB_SIZE    = 13
    GEN_N_CLASSES     = 13
    USE_LONGFORMER    = True

    # Discriminator architecture (must match finetuned checkpoint)
    DISC_EMBED_DIM     = 256
    DISC_NUM_HEADS     = 2
    DISC_NUM_LAYERS    = 2
    DISC_FF_DIM        = 512
    DISC_VOCAB_SIZE    = 12

    # Training
    BATCH_SIZE    = 32
    LR            = 3e-5
    SEED          = 10020
    DEVICE        = "cuda:0"
    EPOCHS        = 3
    ACCUM_STEPS   = 8
    PATIENCE      = 10
    ALPHA         = 0.5      # generation loss weight
    BETA          = 0.7      # specificity loss weight
    LAMBDA        = 0.0      # off-target penalty
    TAU           = 0.5      # soft embedding temperature
    K_NEG         = 5       # negative mRNAs per sample
    START_EPOCH   = 0       # start from epoch 0 if not resuming

    # ── Paths ─────────────────────────────────────────────────────────────
    GEN_CKPT = os.path.join(
        PROJ_HOME,
        "checkpoints/TargetScan/TwoTowerTransformer/CNN-tokenized/TargetGeneration/",
        "520/full_cross_attn/best_token_accuracy_0.9376_epoch18.pth"
    )
    DISC_CKPT = os.path.join(
        PROJ_HOME,
        "checkpoints/TargetScan/TwoTowerTransformer/CNN-tokenized/Manakov2022_train/50/",
        "best_binding_aps_0.8357_epoch14.pth"
    )
    RESUME_CKPT = None
    ECLIP_DATA = os.path.join(PROJ_HOME, "Manakov2022", "AGO2_eCLIP_Manakov2022_train.tsv.gz")

    # ── 1. Build & load Generator ─────────────────────────────────────────
    gen_model = TargetGenerationModel(
        mrna_max_len=MRNA_MAX_LEN, mirna_max_len=MIRNA_MAX_LEN,
        embed_dim=GEN_EMBED_DIM, num_heads=GEN_NUM_HEADS,
        num_layers=GEN_NUM_LAYERS, ff_dim=GEN_FF_DIM,
        batch_size=BATCH_SIZE, vocab_size=GEN_VOCAB_SIZE,
        n_classes=GEN_N_CLASSES, lr=LR, seed=SEED, device=DEVICE,
        use_longformer=USE_LONGFORMER,
    )
    gen_ckpt = torch.load(GEN_CKPT, map_location=DEVICE)
    model_sd = gen_model.state_dict()
    filtered = {k: v for k, v in gen_ckpt.items()
                if k in model_sd and v.shape == model_sd[k].shape}
    gen_model.load_state_dict(filtered, strict=False)
    print(f"Loaded Generator from {GEN_CKPT}  "
          f"({len(filtered)}/{len(gen_ckpt)} tensors, "
          f"{len(gen_ckpt) - len(filtered)} skipped due to shape mismatch)")
    generator = Generator(gen_model)

    # ── 2. Build & load frozen Discriminator ──────────────────────────────
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

    # ── 3. Prepare data ──────────────────────────────────────────────────
    tokenizer = gen_model.tokenizer
    df = pd.read_csv(ECLIP_DATA, compression="gzip", sep="\t")

    df_pos = df[df["label"] == 1].copy()
    df_neg = df[df["label"] == 0].copy()

    from sklearn.model_selection import train_test_split
    df_pos_train, df_pos_val = train_test_split(
        df_pos, test_size=0.15, random_state=SEED,
    )

    ds_train = SpecificityDataset(
        df_pos_train, df_neg, tokenizer,
        mrna_max_len=MRNA_MAX_LEN,
        mirna_max_len=MIRNA_MAX_LEN,
        K=K_NEG,
    )
    ds_val = SpecificityDataset(
        df_pos_val, df_neg, tokenizer,
        mrna_max_len=MRNA_MAX_LEN,
        mirna_max_len=MIRNA_MAX_LEN,
        K=K_NEG,
    )
    train_loader = DataLoader(
        ds_train, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        ds_val, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    print(f"Train: {len(ds_train):,}  Val: {len(ds_val):,}")

    n_train = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    n_frozen_g = sum(p.numel() for p in generator.parameters() if not p.requires_grad)
    n_frozen_d = sum(p.numel() for p in discriminator.parameters())
    print(f"Generator:     {n_train:,} trainable, {n_frozen_g:,} frozen")
    print(f"Discriminator: {n_frozen_d:,} frozen (all)\n")

    # ── 4. Train ──────────────────────────────────────────────────────────
    trainer = SpecificityTrainer(
        generator=generator,
        discriminator=discriminator,
        device=DEVICE,
        lr=LR,
        alpha=ALPHA,
        beta=BETA,
        lam=LAMBDA,
        tau=TAU,
        seed=SEED,
    )
    best = trainer.run(
        train_loader, val_loader,
        epochs=EPOCHS, accumulation_step=ACCUM_STEPS, patience=PATIENCE,
        save_dir=os.path.join(PROJ_HOME, "checkpoints", "specificity_gen", "Manakov2022_train", "ablation_study", "alpha_0.5_beta_0.7_lambda_0.0"),
        wandb_run_name="80nt-Finetune-Specificity-discriminator-on-Manakov2022_train",
        resume_path=RESUME_CKPT,
        start_epoch = START_EPOCH,
        tags = ["ablation_study", "alpha_0.5", "beta_0.7", "lambda_0.0"]
    )
    print(f"\nDone. Best specificity loss = {best:.4f}")

if __name__ == "__main__":
    main()