"""
Finetune Discriminator (CrossAttentionPredictor) on Hard Dataset B
==================================================================

Phase 1 of the two-phase specificity-guided miRNA generation pipeline.

Freeze:   sn_embedding, cnn_embedding, mrna_encoder, mirna_encoder
Finetune: cross_attn_layer, cross_norm, binding_output

Uses differential learning rates and warmup + cosine annealing scheduler.

After training, evaluates discriminator quality:
  AUROC > 0.80  →  good to use as frozen scorer in Phase 2
  AUROC 0.70-0.80  →  usable but increase α (gen_loss weight) in Phase 2
  AUROC < 0.70  →  discriminator too weak, improve before proceeding
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
)

from DTEA_model import CrossAttentionPredictor
from Data_pipeline import CharacterTokenizer
from Global_parameters import PROJ_HOME


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Dataset
# ══════════════════════════════════════════════════════════════════════════════

class eCLIPBindingDataset(Dataset):
    """
    Returns the keys that CrossAttentionPredictor expects:
      mirna_input_ids, mirna_attention_mask,
      mrna_input_ids,  mrna_attention_mask,
      target (label)
    """

    def __init__(self, df, tokenizer, mrna_max_len, mirna_max_len):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.mrna_max_len = mrna_max_len
        self.mirna_max_len = mirna_max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        mrna_seq = row["gene"]
        mirna_seq = row["noncodingRNA"]
        mirna_seq = mirna_seq.replace("U", "T")[::-1]
        label = row["label"]

        mrna_enc = self.tokenizer(
            mrna_seq, 
            padding="max_length", 
            truncation=True,
            max_length=self.mrna_max_len, 
            return_tensors="pt",
        )
        mirna_enc = self.tokenizer(
            mirna_seq, 
            padding="max_length", 
            truncation=True,
            max_length=self.mirna_max_len, 
            return_tensors="pt",
        )

        return {
            "mrna_input_ids":       mrna_enc["input_ids"].squeeze(0),
            "mrna_attention_mask":  mrna_enc["attention_mask"].squeeze(0),
            "mirna_input_ids":      mirna_enc["input_ids"].squeeze(0),
            "mirna_attention_mask": mirna_enc["attention_mask"].squeeze(0),
            "target":               torch.tensor([label], dtype=torch.float),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Freeze / parameter group utilities
# ══════════════════════════════════════════════════════════════════════════════

def freeze_encoders(model: CrossAttentionPredictor):
    """Freeze embedding layers and unused heads."""
    frozen_prefixes = (
        "sn_embedding.",
        "cnn_embedding.",
        "mrna_encoder.",
        "mirna_encoder.",
        "ln_merge.",
        "qa_outputs.",
        "cleavage_head.",
    )
    n_frozen = 0
    for name, param in model.named_parameters():
        if name.startswith(frozen_prefixes):
            param.requires_grad = False
            n_frozen += 1
    return n_frozen


def build_param_groups(
        model: CrossAttentionPredictor,
        lr_slow: float,
        lr_fast: float,
    ):
    """
    Differential learning rates (3 groups):
      slow:    cross_attn_layer, cross_norm  (pretrained, gentle adaptation)
      fast:    binding_output                (decision boundary retraining)
    """
    encoder_prefixes = ("mrna_encoder.", "mirna_encoder.")
    slow_prefixes = ("cross_attn_layer.", "cross_norm.", "dropout.")
    fast_prefixes = ("binding_output.",)

    encoder_params, slow_params, fast_params = [], [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(encoder_prefixes):
            encoder_params.append(param)
        elif name.startswith(slow_prefixes):
            slow_params.append(param)
        elif name.startswith(fast_prefixes):
            fast_params.append(param)
        else:
            slow_params.append(param)

    param_groups = [
        {"params": slow_params, "lr": lr_slow, "label": "cross_attn (slow)"},
        {"params": fast_params, "lr": lr_fast, "label": "binding_output (fast)"},
    ]

    print(f"  Param groups:")
    print(f"    slow    ({lr_slow:.1e}): {sum(p.numel() for p in slow_params):,} params")
    print(f"    fast    ({lr_fast:.1e}): {sum(p.numel() for p in fast_params):,} params")

    return param_groups


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Build scheduler: warmup → cosine annealing
# ══════════════════════════════════════════════════════════════════════════════

def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """
    Linear warmup from 0 → base_lr over warmup_steps,
    then cosine decay to 0 over remaining steps.
    """
    warmup = LinearLR(
        optimizer,
        start_factor=1e-2,   # start at 1% of base lr
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=1e-5,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )
    return scheduler


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Trainer
# ══════════════════════════════════════════════════════════════════════════════

class DiscriminatorTrainer:
    def __init__(
        self,
        model: CrossAttentionPredictor,
        device: str = "cuda",
        lr_slow: float = 5e-5,
        lr_fast: float = 1e-4,
        warmup_fraction: float = 0.05,
        max_grad_norm: float = 1.0,
        seed: int = 42,
    ):
        self.model = model
        self.device = torch.device(device)
        self.max_grad_norm = max_grad_norm
        self.lr_slow = lr_slow
        self.lr_fast = lr_fast
        self.warmup_fraction = warmup_fraction
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Freeze unused layers
        n_frozen = freeze_encoders(model)
        n_total = sum(1 for _ in model.parameters())
        n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        print(f"Parameters: {n_frozen} frozen, {n_trainable} trainable (of {n_total} total)")

        # Differential learning rates (2 groups)
        param_groups = build_param_groups(model, lr_slow, lr_fast)
        self.optimizer = AdamW(param_groups, weight_decay=0.01)

        self._seed_everything(seed)

    @staticmethod
    def _seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _step(self, batch):
        """Single forward pass, returns loss and predictions."""
        mirna_ids  = batch["mirna_input_ids"].to(self.device)
        mrna_ids   = batch["mrna_input_ids"].to(self.device)
        mirna_mask = batch["mirna_attention_mask"].to(self.device)
        mrna_mask  = batch["mrna_attention_mask"].to(self.device)
        targets    = batch["target"].to(self.device)

        outputs = self.model(
            mirna=mirna_ids,
            mrna=mrna_ids,
            mrna_mask=mrna_mask,
            mirna_mask=mirna_mask,
        )
        binding_logit = outputs[0]  # (B, 1)
        loss = self.loss_fn(binding_logit.squeeze(-1), targets.view(-1).float())

        return loss, binding_logit.squeeze(-1), targets

    def train_epoch(self, dataloader, epoch, scheduler, accumulation_step=1):
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()
        loss_buf = []

        for i, batch in enumerate(dataloader):
            loss, _, _ = self._step(batch)
            (loss / accumulation_step).backward()

            loss_buf.append(loss.item())
            total_loss += loss.item()

            if (i + 1) % accumulation_step == 0:
                clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.max_grad_norm,
                )
                self.optimizer.step()
                scheduler.step()
                self.optimizer.zero_grad()

                if (i + 1) % (accumulation_step * 50) == 0:
                    bs = batch["mrna_input_ids"].size(0)
                    avg = sum(loss_buf) / len(loss_buf)
                    current_lrs = [f"{g['lr']:.2e}" for g in self.optimizer.param_groups]
                    print(
                        f"  Epoch {epoch} "
                        f"[{(i+1)*bs}/{len(dataloader.dataset)} "
                        f"({(i+1)*bs/len(dataloader.dataset)*100:.0f}%)] "
                        f"loss={avg:.4f}  lr={current_lrs}",
                        flush=True,
                    )
                loss_buf = []

        # Flush remaining gradients
        if (i + 1) % accumulation_step != 0:
            clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.max_grad_norm,
            )
            self.optimizer.step()
            scheduler.step()
            self.optimizer.zero_grad()

        return total_loss / len(dataloader)

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        all_labels, all_probs = [], []

        for batch in dataloader:
            loss, logits, targets = self._step(batch)
            total_loss += loss.item()
            all_probs.append(torch.sigmoid(logits).cpu()) 
            all_labels.append(targets.cpu())

        y = torch.cat(all_labels).numpy()
        p = torch.cat(all_probs).numpy()
        yhat = (p >= 0.5).astype(int)
        n = len(dataloader)

        return {
            "loss":      total_loss / n,
            "accuracy":  accuracy_score(y, yhat),
            "f1":        f1_score(y, yhat),
            "precision": precision_score(y, yhat),
            "recall":    recall_score(y, yhat),
            "auroc":     roc_auc_score(y, p),
            "aps":       average_precision_score(y, p),
            "mcc":       matthews_corrcoef(y, yhat),
            
        }

    def run(
        self,
        train_loader,
        val_loader,
        mode: str = "train",
        epochs: int = 20,
        accumulation_step: int = 4,
        patience: int = 5,
        save_dir: str = "checkpoints/discriminator_finetuned",
        wandb_project: str = "Finetune_discriminator",
        wandb_run_name: str | None = None,
    ):
        if mode == "evaluate":
            self.model.to(self.device)
            m = self.evaluate(val_loader)
            print(
                f"\n{'═'*60}\n"
                f"  Evaluation Results\n"
                f"  Val loss:   {m['loss']:.4f}\n"
                f"  Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}\n"
                f"  Prec={m['precision']:.4f}  Rec={m['recall']:.4f}\n"
                f"  AUROC={m['auroc']:.4f}  APS={m['aps']:.4f}\n"
                f"  MCC={m['mcc']:.4f}\n"
                f"{'═'*60}\n",
                flush=True,
            )
            return m["aps"]
        elif mode == "train":
            os.makedirs(save_dir, exist_ok=True)
            self.model.to(self.device)

            # ── W&B init ──────────────────────────────────────────────────
            wandb.login(key="your key")
            run = wandb.init(
                project=wandb_project,
                name=wandb_run_name or "500nt_discriminator-finetune-eCLIP",
                config={
                    "epochs": epochs,
                    "batch_size": train_loader.batch_size,
                    "effective_batch_size": train_loader.batch_size * accumulation_step,
                    "lr_slow": self.lr_slow,
                    "lr_fast": self.lr_fast,
                    "warmup_fraction": self.warmup_fraction,
                    "max_grad_norm": self.max_grad_norm,
                    "patience": patience,
                    "train_samples": len(train_loader.dataset),
                    "val_samples": len(val_loader.dataset),
                },
                tags=["discriminator", "finetune", "eCLIP", "binding"],
                save_code=False,
                job_type="train",
            )

            # Build scheduler
            steps_per_epoch = len(train_loader) // accumulation_step
            total_steps = steps_per_epoch * epochs
            warmup_steps = int(total_steps * self.warmup_fraction)
            scheduler = build_scheduler(self.optimizer, warmup_steps, total_steps)
            print(f"Scheduler: {warmup_steps} warmup steps, {total_steps} total steps\n")

            best_auroc = 0.0
            counter = 0

            for epoch in range(epochs):
                train_loss = self.train_epoch(
                    train_loader, epoch, scheduler, accumulation_step,
                )
                m = self.evaluate(val_loader)

                print(
                    f"\n{'═'*60}\n"
                    f"  Epoch {epoch}\n"
                    f"  Train loss: {train_loss:.4f}\n"
                    f"  Val loss:   {m['loss']:.4f}\n"
                    f"  Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}\n"
                    f"  Prec={m['precision']:.4f}  Rec={m['recall']:.4f}\n"
                    f"  AUROC={m['auroc']:.4f}  APS={m['aps']:.4f}\n"
                    f"  MCC={m['mcc']:.4f}\n"
                    f"{'═'*60}\n",
                    flush=True,
                )

                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "eval/loss": m["loss"],
                    "eval/accuracy": m["accuracy"],
                    "eval/f1": m["f1"],
                    "eval/precision": m["precision"],
                    "eval/recall": m["recall"],
                    "eval/auroc": m["auroc"],
                    "eval/aps": m["aps"],
                    "eval/mcc": m["mcc"],
                }, step=epoch)

                if m["auroc"] > best_auroc:
                    best_auroc = m["auroc"]
                    counter = 0
                    path = os.path.join(
                        save_dir, f"best_auroc_{best_auroc:.4f}_epoch{epoch}.pth",
                    )
                    torch.save(self.model.state_dict(), path)
                    print(f"  ★ Saved → {path}")
                else:
                    counter += 1
                    if counter >= patience:
                        print(f"  Early stopping at epoch {epoch}")
                        break

            # ── Quality gate ──────────────────────────────────────────────
            print(f"\n{'='*60}")
            print(f"  QUALITY GATE")
            print(f"  Best AUROC: {best_auroc:.4f}")
            if best_auroc >= 0.80:
                print(f"  ✓ PASS — discriminator is ready for Phase 2")
                print(f"    Recommended: freeze D, train generator with α=0.3, β=1.0")
            elif best_auroc >= 0.70:
                print(f"  △ MARGINAL — usable but discriminator signal will be noisy")
                print(f"    Recommended: increase α (gen_loss) in Phase 2, e.g. α=0.5, β=1.0")
            else:
                print(f"  ✗ FAIL — discriminator cannot reliably distinguish hard pairs")
                print(f"    Recommended: improve D before proceeding to Phase 2")
                print(f"    Options: more data, unfreeze mirna_encoder, harder negatives")
            print(f"{'='*60}\n")

            wandb.finish()
            return best_auroc


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Checkpoint loading utility
# ══════════════════════════════════════════════════════════════════════════════

def load_dtea_checkpoint(
    model: CrossAttentionPredictor,
    ckpt_path: str,
    device: str = "cpu",
    prefix: str = "predictor.",
):
    """
    Load DTEA checkpoint into CrossAttentionPredictor.
    DTEA wraps CrossAttentionPredictor under `self.predictor`,
    so keys are prefixed with 'predictor.'.
    """
    sd = torch.load(ckpt_path, map_location=device)
    model_sd = model.state_dict()

    to_load = {}
    for k, v in sd.items():
        # Strip prefix if present
        if k.startswith(prefix):
            new_k = k[len(prefix):]
        else:
            new_k = k

        if new_k in model_sd:
            if v.shape == model_sd[new_k].shape:
                to_load[new_k] = v
            else:
                print(f"  Shape mismatch, skipping: {new_k} "
                      f"(ckpt {v.shape} vs model {model_sd[new_k].shape})")

    missing, unexpected = model.load_state_dict(to_load, strict=False)
    print(f"Loaded {len(to_load)} tensors from {ckpt_path}")

    # Filter out expected missing (rotary buffers)
    real_missing = [k for k in missing if "rotary" not in k]
    if real_missing:
        print(f"  Missing (non-rotary): {real_missing[:10]}...")
    if unexpected:
        print(f"  Unexpected: {unexpected[:10]}...")

    return model


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Hyperparameters ───────────────────────────────────────────────────
    MRNA_MAX_LEN  = 50      # must be divisible by 2*window_size=40 if using Longformer
    MIRNA_MAX_LEN = 26
    EMBED_DIM     = 256
    NUM_HEADS     = 2
    NUM_LAYERS    = 2
    FF_DIM        = 512
    VOCAB_SIZE    = 12
    BATCH_SIZE    = 32
    SEED          = 10020
    DEVICE        = "cuda:3"
    EPOCHS        = 30
    ACCUM_STEPS   = 8
    PATIENCE      = 5
    MODE          = "evaluate"
    USE_LONGFORMER = False

    # Differential learning rates (3 groups)
    LR_SLOW       = 3e-5    # cross_attn_layer, cross_norm
    LR_FAST       = 1e-4    # binding_output
    WARMUP_FRAC   = 0.01    # 1% of total steps

    # ── Paths (update these) ──────────────────────────────────────────────
    DTEA_CKPT = os.path.join(
        PROJ_HOME,
        "checkpoints/TargetScan/TwoTowerTransformer/CNN-tokenized/Manakov2022_train/50/best_binding_aps_0.8357_epoch14.pth"
    )
    INPUT_DATA = os.path.join(
        PROJ_HOME,
        "AGO2_eCLIP_Manakov2022_leftout.tsv.gz",
    )

    # ── 1. Build model ────────────────────────────────────────────────────
    model = CrossAttentionPredictor(
        mirna_max_len=MIRNA_MAX_LEN,
        mrna_max_len=MRNA_MAX_LEN,
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        hidden_sizes=[FF_DIM, FF_DIM],
        n_classes=1,                # binary binding classification
        dropout_rate=0.2,
        device=DEVICE,
        predict_span=False,         # not needed for binding-only finetuning
        predict_binding=True,
        predict_cleavage=False,
        use_longformer=USE_LONGFORMER,
        window_size=20,
    )

    # ── 2. Load pretrained weights ────────────────────────────────────────
    model = load_dtea_checkpoint(model, DTEA_CKPT, device=DEVICE)

    # ── 3. Prepare data ──────────────────────────────────────────────────
    tokenizer = CharacterTokenizer(
        characters=["A", "T", "C", "G", "N"],
        model_max_length=MRNA_MAX_LEN,
        padding_side="right",
    )

    df = pd.read_csv(INPUT_DATA, compression="gzip", sep="\t")
    print(f"Dataset: {len(df):,} samples, label distribution:\n{df['label'].value_counts()}\n")

    if MODE == "train":
        from sklearn.model_selection import train_test_split
        df_train, df_val = train_test_split(
            df, test_size=0.15, stratify=df["label"], random_state=SEED, shuffle=True,
        )

        ds_train = eCLIPBindingDataset(df_train, tokenizer, MRNA_MAX_LEN, MIRNA_MAX_LEN)
        ds_val   = eCLIPBindingDataset(df_val,   tokenizer, MRNA_MAX_LEN, MIRNA_MAX_LEN)

        train_loader = DataLoader(
            ds_train, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=4, pin_memory=True,
        )
        val_loader = DataLoader(
            ds_val, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=4, pin_memory=True,
        )
        print(f"Train: {len(ds_train):,}  Val: {len(ds_val):,}\n")

    elif MODE == "evaluate":
        ds_val = eCLIPBindingDataset(df, tokenizer, MRNA_MAX_LEN, MIRNA_MAX_LEN)
        val_loader = DataLoader(
            ds_val, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=4, pin_memory=True,
        )
        train_loader = None

    # ── 4. Train ──────────────────────────────────────────────────────────
    trainer = DiscriminatorTrainer(
        model=model,
        device=DEVICE,
        lr_slow=LR_SLOW,
        lr_fast=LR_FAST,
        warmup_fraction=WARMUP_FRAC,
        seed=SEED,
    )

    best_aps = trainer.run(
        train_loader=train_loader,
        val_loader=val_loader,
        mode=MODE,
        epochs=EPOCHS,
        accumulation_step=ACCUM_STEPS,
        patience=PATIENCE,
        save_dir=os.path.join(PROJ_HOME, "checkpoints", "discriminator_finetuned"),
        
    )
    print(f"Done. Best APS = {best_aps:.4f}")
