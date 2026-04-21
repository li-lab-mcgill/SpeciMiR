"""
Phase 2: Specificity-Guided Generator Training via Frozen Discriminator
=======================================================================

Generator (TargetGenerationModel):
  - Encoder: FROZEN
  - Decoder + predictor_head: TRAINABLE

Discriminator (CrossAttentionPredictor):
  - Entirely FROZEN
  - Provides differentiable score D(miRNA, mRNA) via soft embedding bypass

Loss:
  L = α · L_gen + β · L_specificity
  L_specificity = -log D(soft_emb, target_mRNA) + λ · mean[-log(1 - D(soft_emb, neg_mRNA_k))]

Gradient flow:
  loss → D (frozen, no update) → soft_emb → soft_probs → gen_logits → decoder params
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader

from DTEA_model import (
    TargetGenerationModel,
    CrossAttentionPredictor,
)
from Global_parameters import PROJ_HOME


def load_discriminator_checkpoint(model, ckpt_path, device):
    old_sd = torch.load(ckpt_path, map_location=device)
    new_sd = model.state_dict()
    to_load = {}
    # remap old_key -> new key by stripping "predictor."
    def remap_key(k):
        if k.startswith("predictor."):
            return k[len("predictor."):]
        return k
    for sd_key, sd_value in old_sd.items():
        new_key = remap_key(sd_key)
        if new_key in model.state_dict():
            if "rotary.cos_emb" in new_key or "rotary.sin_emb" in new_key:
                if old_sd[sd_key].shape == new_sd[new_key].shape:
                    to_load[new_key] = old_sd[sd_key]
                else:
                    print(f"Shape mismatch, skipping: {new_key} (ckpt {old_sd[sd_key].shape} vs model {new_sd[new_key].shape})")
                    continue
            else:
                if old_sd[sd_key].shape == new_sd[new_key].shape:
                    to_load[new_key] = old_sd[sd_key]
                else:
                    print(f"Shape mismatch, skipping: {new_key} (ckpt {old_sd[sd_key].shape} vs model {new_sd[new_key].shape})")
                    continue
        else:
            print(f"{new_key} not found in checkpoint.")
            continue
    missing, unexpected = model.load_state_dict(to_load, strict=False)
    print(f"Loaded {len(new_sd)} tensors from {ckpt_path}")
    print("missing:", missing)
    print("unexpected:", unexpected)
    return model

# ══════════════════════════════════════════════════════════════════════════════
# 1.  Dataset — each sample has target mRNA, true miRNA, and K negative mRNAs
# ══════════════════════════════════════════════════════════════════════════════

class SpecificityDataset(Dataset):
    """
    For each positive (miRNA, target_mRNA) pair, samples K negative mRNAs
    from the same dataset (mRNAs paired with label=0 for different miRNAs,
    or random mRNAs from the negative pool).

    Returns:
      target_mrna_ids, target_mrna_mask   — the intended target mRNA
      mirna_ids, mirna_mask               — the true miRNA (for teacher forcing)
      neg_mrna_ids, neg_mrna_masks        — (K, L) negative mRNAs
    """

    def __init__(self, df_pos, df_neg_pool, tokenizer, mrna_max_len, mirna_max_len, K=5):
        """
        df_pos:      DataFrame of positive pairs (label=1)
        df_neg_pool: DataFrame to sample negative mRNAs from (can be label=0 rows
                     or all unique mRNAs). Must have a 'gene' column.
        K:           number of negative mRNAs per sample
        """
        self.df_pos = df_pos.reset_index(drop=True)
        self.neg_genes = df_neg_pool["gene"].unique().tolist()
        self.tokenizer = tokenizer
        self.mrna_max_len = mrna_max_len
        self.mirna_max_len = mirna_max_len
        self.K = K

    def __len__(self):
        return len(self.df_pos)

    def _encode_seq(self, seq, max_len):
        enc = self.tokenizer(
            str(seq), padding="max_length", truncation=True,
            max_length=max_len, return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

    def __getitem__(self, idx):
        row = self.df_pos.iloc[idx]

        # Target mRNA
        target_ids, target_mask = self._encode_seq(row["gene"], self.mrna_max_len)

        # True miRNA (for teacher forcing)
        mirna_ids, mirna_mask = self._encode_seq(row["noncodingRNA"], self.mirna_max_len)

        # K negative mRNAs (random from pool, excluding the target)
        target_gene = str(row["gene"])
        neg_ids_list, neg_mask_list = [], []
        candidates = [g for g in random.sample(self.neg_genes, min(self.K * 2, len(self.neg_genes)))
                       if g != target_gene][:self.K]
        # pad if not enough candidates
        while len(candidates) < self.K:
            candidates.append(random.choice(self.neg_genes))

        for neg_gene in candidates:
            nid, nmask = self._encode_seq(neg_gene, self.mrna_max_len)
            neg_ids_list.append(nid)
            neg_mask_list.append(nmask)

        neg_mrna_ids   = torch.stack(neg_ids_list)    # (K, L_mrna)
        neg_mrna_masks = torch.stack(neg_mask_list)    # (K, L_mrna)

        return {
            "target_mrna_ids":   target_ids,
            "target_mrna_mask":  target_mask,
            "mirna_ids":         mirna_ids,
            "mirna_mask":        mirna_mask,
            "neg_mrna_ids":      neg_mrna_ids,
            "neg_mrna_masks":    neg_mrna_masks,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Discriminator wrapper — adds forward_from_embedding
# ══════════════════════════════════════════════════════════════════════════════

class FrozenDiscriminator(nn.Module):
    """
    Wraps a pretrained CrossAttentionPredictor.
    Entirely frozen. Provides `score_from_soft_embedding` for gradient pass-through.
    """

    def __init__(self, predictor: CrossAttentionPredictor):
        super().__init__()
        self.predictor = predictor
        # Freeze everything
        for p in self.predictor.parameters():
            p.requires_grad = False
        self.predictor.eval()

    @property
    def sn_embedding_weight(self):
        """The embedding matrix for computing soft embeddings."""
        return self.predictor.sn_embedding.weight  # (V, D)

    def score_from_soft_embedding(self, mirna_soft_emb, mrna_ids, mrna_mask, mirna_mask):
        """
        Score miRNA–mRNA interaction using continuous miRNA soft embeddings.

        mirna_soft_emb : (B, L_mirna, D) — from generator's softmax @ embedding
        mrna_ids       : (B, L_mrna)     — discrete token IDs
        mrna_mask      : (B, L_mrna)     — 1=valid, 0=pad
        mirna_mask     : (B, L_mirna)    — 1=valid, 0=pad

        Returns: (B,) interaction probability (after sigmoid)
        """
        p = self.predictor

        # ── mRNA path: normal discrete processing ─────────────────────
        mrna_sn = p.sn_embedding(mrna_ids)
        mrna_cnn = p.cnn_embedding(mrna_sn.transpose(-1, -2))
        mrna_emb = mrna_sn + mrna_cnn

        if p.use_longformer:
            lf_mask = torch.where(
                mrna_mask > 0,
                torch.zeros_like(mrna_mask),
                torch.full_like(mrna_mask, fill_value=-1),
            )
            mrna_emb = p.mrna_encoder(mrna_emb, mask=lf_mask)
        else:
            mrna_emb = p.mrna_encoder(mrna_emb, mask=mrna_mask)

        # ── miRNA path: soft embedding bypass ─────────────────────────
        # mirna_soft_emb replaces p.sn_embedding(mirna_ids)
        # Still apply CNN + encoder on top (frozen, but differentiable)
        mirna_cnn = p.cnn_embedding(mirna_soft_emb.transpose(-1, -2))
        mirna_emb = mirna_soft_emb + mirna_cnn
        mirna_emb = p.mirna_encoder(mirna_emb, mask=mirna_mask)

        # ── cross-attention + binding head ────────────────────────────
        if p.use_longformer:
            z = p.cross_attn_layer(
                query=mrna_emb, key=mirna_emb, value=mirna_emb,
                attention_mask=mirna_mask,
                query_attention_mask=mrna_mask,
            )[0]
        else:
            z = p.cross_attn_layer(
                query=mrna_emb, key=mirna_emb, value=mirna_emb,
                mask=mirna_mask,
            )

        z_res = p.dropout(z) + mrna_emb
        z_norm = p.cross_norm(z_res)
        z_norm = z_norm.masked_fill(mrna_mask.unsqueeze(-1) == 0, 0)

        valid_counts = mrna_mask.sum(dim=1, keepdim=True)
        pooled = z_norm.sum(dim=1) / (valid_counts + 1e-8)
        logit = p.binding_output(pooled).squeeze(-1)  # (B,)
        return torch.sigmoid(logit)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Generator wrapper — produces soft embeddings
# ══════════════════════════════════════════════════════════════════════════════

class Generator(nn.Module):
    """
    Wraps TargetGenerationModel.
    Encoder: frozen. Decoder + predictor_head: trainable.
    """

    def __init__(self, gen_model: TargetGenerationModel):
        super().__init__()
        self.gen_model = gen_model
        self._freeze_encoder()

    def _freeze_encoder(self):
        for name, param in self.gen_model.named_parameters():
            if name.startswith(("sn_embedding.", "cnn_embedding.", "mrna_encoder.")):
                param.requires_grad = False

    @property
    def pad_idx(self):
        return self.gen_model.pad_idx

    @property
    def device(self):
        return next(self.parameters()).device

    def create_src_mask(self, src):
        return (src != self.pad_idx).to(torch.uint8)

    def create_tgt_mask_causal(self, tgt):
        B, L = tgt.size()
        causal = torch.tril(torch.ones(L, L, dtype=torch.uint8, device=tgt.device)).unsqueeze(0)
        non_pad = (tgt != self.pad_idx).to(torch.uint8).unsqueeze(1).expand(B, L, L)
        return causal & non_pad

    def create_tgt_mask_1d(self, tgt):
        return (tgt != self.pad_idx).to(torch.uint8)

    def forward_teacher_forcing(self, mrna_ids, mirna_ids):
        """
        Standard teacher-forcing forward pass.
        Returns gen_logits (B, L_tgt, V) for computing generation loss.
        """
        m = self.gen_model
        tgt_input = mirna_ids[:, :-1]
        src_mask = self.create_src_mask(mrna_ids).to(mrna_ids.device)
        tgt_mask = self.create_tgt_mask_causal(tgt_input).to(mrna_ids.device)

        logits = m(
            mirna=tgt_input,
            mrna=mrna_ids,
            mrna_mask=src_mask,
            mirna_mask=tgt_mask,
        )
        return logits, tgt_input

    def logits_to_soft_embedding(self, logits, disc_embedding_weight, tau=0.5):
        # logits: (B, L, 13) — index 12 is global attn token, never generated
        # disc_embedding_weight: (12, D)

        logits_shared = logits[:, :, :12]                       # drop global token logit
        soft_probs = F.softmax(logits_shared / tau, dim=-1)     # (B, L, 12)
        soft_emb = soft_probs @ disc_embedding_weight            # (B, L, D)
        return soft_emb


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Trainer
# ══════════════════════════════════════════════════════════════════════════════

class SpecificityTrainer:
    def __init__(
        self,
        generator: Generator,
        discriminator: FrozenDiscriminator,
        device: str = "cuda",
        lr: float = 1e-5,
        alpha: float = 0.5,      # generation loss weight
        beta: float = 0.7,       # specificity loss weight
        lam: float = 0.5,        # off-target penalty within specificity loss
        tau: float = 0.5,        # soft embedding temperature
        warmup_fraction: float = 0.05,
        max_grad_norm: float = 1.0,
        seed: int = 42,
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.device = torch.device(device)
        self.alpha = alpha
        self.beta = beta
        self.lam = lam
        self.tau = tau
        self.warmup_fraction = warmup_fraction
        self.max_grad_norm = max_grad_norm

        self.gen_loss_fn = nn.CrossEntropyLoss(ignore_index=generator.pad_idx)

        trainable = [p for p in generator.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable, lr=lr)
        self._seed_everything(seed)

    @staticmethod
    def _seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _compute_specificity_loss(self, soft_emb, mirna_mask_1d, batch):
        """
        Compute:
          L_spec = -log D(soft_emb, target) + λ · mean_k[-log(1 - D(soft_emb, neg_k))]
        """
        D = self.discriminator
        target_ids  = batch["target_mrna_ids"].to(self.device)
        target_mask = batch["target_mrna_mask"].to(self.device)
        neg_ids     = batch["neg_mrna_ids"].to(self.device)     # (B, K, L)
        neg_masks   = batch["neg_mrna_masks"].to(self.device)   # (B, K, L)

        # On-target score (want HIGH)
        p_on = D.score_from_soft_embedding(
            soft_emb, target_ids, target_mask, mirna_mask_1d,
        )  # (B,)
        on_target_loss = -torch.log(p_on + 1e-8).mean()

        # Off-target scores (want LOW)
        B, K, L = neg_ids.shape
        off_target_losses = []
        off_target_scores = []
        for k in range(K):
            p_off_k = D.score_from_soft_embedding(
                soft_emb, neg_ids[:, k], neg_masks[:, k], mirna_mask_1d,
            )  # (B,)
            off_target_losses.append(-torch.log(1 - p_off_k + 1e-8).mean())
            off_target_scores.append(p_off_k.mean().item())

        off_target_loss = torch.stack(off_target_losses).mean()
        loss = on_target_loss + self.lam * off_target_loss

        return loss, p_on.mean().item(), sum(off_target_scores) / len(off_target_scores)

    def train_epoch(self, dataloader, epoch, accumulation_step=1, scheduler=None):
        self.generator.train()
        self.discriminator.eval()

        total_gen_loss, total_spec_loss, total_loss = 0., 0., 0.
        total_p_on, total_p_off = 0., 0.
        self.optimizer.zero_grad()
        loss_buf = []

        emb_weight = self.discriminator.sn_embedding_weight  # (V, D)

        for i, batch in enumerate(dataloader):
            mrna_ids  = batch["target_mrna_ids"].to(self.device)
            mirna_ids = batch["mirna_ids"].to(self.device)

            # ── 1. Teacher forcing → generation loss ──────────────────
            gen_logits, tgt_input = self.generator.forward_teacher_forcing(
                mrna_ids, mirna_ids,
            )
            tgt_output = mirna_ids[:, 1:]
            B, L, V = gen_logits.size()
            gen_loss = self.gen_loss_fn(
                gen_logits.view(B * L, V), tgt_output.reshape(B * L),
            )

            # ── 2. Soft embedding → specificity loss ──────────────────
            soft_emb = self.generator.logits_to_soft_embedding(
                gen_logits, emb_weight, self.tau,
            )  # (B, L_tgt, D)

            mirna_mask_1d = self.generator.create_tgt_mask_1d(tgt_input).to(self.device)

            spec_loss, p_on, p_off = self._compute_specificity_loss(
                soft_emb, mirna_mask_1d, batch,
            )

            # ── 3. Total loss ─────────────────────────────────────────
            loss = self.alpha * gen_loss + self.beta * spec_loss

            (loss / accumulation_step).backward()
            loss_buf.append(loss.item())
            total_gen_loss  += gen_loss.item()
            total_spec_loss += spec_loss.item()
            total_loss      += loss.item()
            total_p_on      += p_on
            total_p_off     += p_off

            if (i + 1) % accumulation_step == 0:
                clip_grad_norm_(
                    [p for p in self.generator.parameters() if p.requires_grad],
                    self.max_grad_norm,
                )
                self.optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                self.optimizer.zero_grad()

                if (i + 1) % (accumulation_step * 20) == 0:
                    bs = mrna_ids.size(0)
                    n_seen = (i + 1) * bs
                    n_total = len(dataloader.dataset)
                    avg = sum(loss_buf) / len(loss_buf)
                    print(
                        f"  Epoch {epoch} [{n_seen}/{n_total} ({n_seen/n_total*100:.0f}%)] "
                        f"loss={avg:.4f} gen={gen_loss.item():.4f} spec={spec_loss.item():.4f} "
                        f"p_on={p_on:.3f} p_off={p_off:.3f}",
                        flush=True,
                    )
                loss_buf = []

        # Flush
        if (i + 1) % accumulation_step != 0:
            clip_grad_norm_(
                [p for p in self.generator.parameters() if p.requires_grad],
                self.max_grad_norm,
            )
            self.optimizer.step()
            if scheduler is not None:
                scheduler.step()
            self.optimizer.zero_grad()

        n = len(dataloader)
        return {
            "loss":      total_loss / n,
            "gen_loss":  total_gen_loss / n,
            "spec_loss": total_spec_loss / n,
            "p_on":      total_p_on / n,
            "p_off":     total_p_off / n,
        }

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.generator.eval()
        self.discriminator.eval()
        emb_weight = self.discriminator.sn_embedding_weight

        total_loss, total_gen, total_spec = 0., 0., 0.
        total_p_on, total_p_off = 0., 0.

        for batch in dataloader:
            mrna_ids  = batch["target_mrna_ids"].to(self.device)
            mirna_ids = batch["mirna_ids"].to(self.device)

            gen_logits, tgt_input = self.generator.forward_teacher_forcing(
                mrna_ids, mirna_ids,
            )
            tgt_output = mirna_ids[:, 1:]
            B, L, V = gen_logits.size()
            gen_loss = self.gen_loss_fn(
                gen_logits.view(B * L, V), tgt_output.reshape(B * L),
            )

            soft_emb = self.generator.logits_to_soft_embedding(
                gen_logits, emb_weight, self.tau,
            )
            mirna_mask_1d = self.generator.create_tgt_mask_1d(tgt_input).to(self.device)
            spec_loss, p_on, p_off = self._compute_specificity_loss(
                soft_emb, mirna_mask_1d, batch,
            )

            total_gen  += gen_loss.item()
            total_spec += spec_loss.item()
            total_loss += self.alpha * gen_loss.item() + self.beta * spec_loss.item()
            total_p_on  += p_on
            total_p_off += p_off

        n = len(dataloader)
        return {
            "gen_loss":  total_gen / n,
            "spec_loss": total_spec / n,
            "loss":      total_loss / n,
            "p_on":      total_p_on / n,
            "p_off":     total_p_off / n,
        }

    def run(
        self,
        train_loader,
        val_loader,
        epochs=20,
        accumulation_step=4,
        patience=5,
        save_dir="checkpoints/specificity_gen",
        wandb_project: str = "Finetune_Specificity",
        wandb_run_name: str | None = None,
        resume_path: str | None = None,
        start_epoch: int = 0,
        tags: list[str] = [],
    ):
        os.makedirs(save_dir, exist_ok=True)
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        # ── W&B init ──────────────────────────────────────────────────
        wandb.login(key="600e5cca820a9fbb7580d052801b3acfd5c92da2")
        wandb.init(
            project=wandb_project,
            name=wandb_run_name or "specificity-gen-finetune",
            config={
                "epochs": epochs,
                "batch_size": train_loader.batch_size,
                "effective_batch_size": train_loader.batch_size * accumulation_step,
                "lr": self.optimizer.defaults["lr"],
                "alpha": self.alpha,
                "beta": self.beta,
                "lambda": self.lam,
                "tau": self.tau,
                "warmup_fraction": self.warmup_fraction,
                "max_grad_norm": self.max_grad_norm,
                "patience": patience,
                "train_samples": len(train_loader.dataset),
                "val_samples": len(val_loader.dataset),
                "resumed_from": resume_path or "none",
            },
            tags=["generator", "specificity", "Manakov2022_train"] + tags,
            save_code=False,
            job_type="train",
        )

        # Scheduler
        steps_per_epoch = len(train_loader) // accumulation_step
        total_steps = steps_per_epoch * epochs
        warmup_steps = int(total_steps * self.warmup_fraction)
        warmup = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(self.optimizer, T_max=total_steps - warmup_steps, eta_min=1e-7)
        scheduler = SequentialLR(self.optimizer, [warmup, cosine], milestones=[warmup_steps])

        # ── Resume from checkpoint ────────────────────────────────────
        best_loss = float("inf")
        counter = 0

        if resume_path is not None and os.path.exists(resume_path):
            ckpt = torch.load(resume_path, map_location=self.device)
            if isinstance(ckpt, dict) and "generator_state_dict" in ckpt:
                self.generator.gen_model.load_state_dict(ckpt["generator_state_dict"])
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                start_epoch = ckpt["epoch"] + 1
                best_loss = ckpt.get("best_loss", float("inf"))
                counter = ckpt.get("patience_counter", 0)
                print(f"Resumed full training state from {resume_path} (epoch {ckpt['epoch']})")
            else:
                missing, unexpected = self.generator.gen_model.load_state_dict(ckpt, strict=False)
                print(f"Resumed model weights from {resume_path}")
                if missing:
                    print(f"  Missing: {missing}")
                if unexpected:
                    print(f"  Unexpected: {unexpected}")
                # Advance scheduler to correct position
                skip_steps = start_epoch * steps_per_epoch
                for _ in range(skip_steps):
                    scheduler.step()
                print(f"  Advanced scheduler by {skip_steps} steps to epoch {start_epoch}")

        print(f"Scheduler: {warmup_steps} warmup, {total_steps} total steps")
        print(f"Training epochs {start_epoch}..{epochs - 1}\n")

        for epoch in range(start_epoch, epochs):
            tm = self.train_epoch(train_loader, epoch, accumulation_step, scheduler)
            vm = self.evaluate(val_loader)

            print(
                f"\n{'═'*65}\n"
                f"  Epoch {epoch}\n"
                f"  Train: loss={tm['loss']:.4f}  gen={tm['gen_loss']:.4f}  "
                f"spec={tm['spec_loss']:.4f}  p_on={tm['p_on']:.3f}  p_off={tm['p_off']:.3f}\n"
                f"  Val:   loss={vm['loss']:.4f}  gen={vm['gen_loss']:.4f}  spec={vm['spec_loss']:.4f}  "
                f"p_on={vm['p_on']:.3f}  p_off={vm['p_off']:.3f}\n"
                f"{'═'*65}\n",
                flush=True,
            )

            wandb.log({
                "epoch": epoch,
                "train/loss": tm["loss"],
                "train/gen_loss": tm["gen_loss"],
                "train/spec_loss": tm["spec_loss"],
                "train/p_on": tm["p_on"],
                "train/p_off": tm["p_off"],
                "eval/loss": vm["loss"],
                "eval/gen_loss": vm["gen_loss"],
                "eval/spec_loss": vm["spec_loss"],
                "eval/p_on": vm["p_on"],
                "eval/p_off": vm["p_off"],
                "eval/specificity_gap": vm["p_on"] - vm["p_off"],
            }, step=epoch)

            # Track val specificity loss (lower = better on-target, less off-target)
            if vm["loss"] < best_loss:
                best_loss = vm["loss"]
                counter = 0
                path = os.path.join(save_dir, f"best_loss_{best_loss:.4f}_epoch{epoch}.pth")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save({
                    "generator_state_dict": self.generator.gen_model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "patience_counter": counter,
                }, path)
                print(f"  ★ Saved → {path}")
            else:
                counter += 1
                if counter >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        wandb.finish()
        return best_loss


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
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
    EPOCHS        = 9
    ACCUM_STEPS   = 8
    PATIENCE      = 10
    ALPHA         = 0.5      # generation loss weight
    BETA          = 0.7      # specificity loss weight
    LAMBDA        = 0.5      # off-target penalty
    TAU           = 0.5      # soft embedding temperature
    K_NEG         = 5       # negative mRNAs per sample
    START_EPOCH   = 1       # start from epoch 0 if not resuming

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
    RESUME_CKPT = os.path.join(
        PROJ_HOME,
        "checkpoints/specificity_gen/Manakov2022_train/",
        "best_spec_0.6128_epoch0.pth"
    )
    ECLIP_DATA = os.path.join(PROJ_HOME, "AGO2_eCLIP_Manakov2022_train.tsv.gz")

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
        save_dir=os.path.join(PROJ_HOME, "checkpoints", "specificity_gen", "Manakov2022_train"),
        wandb_run_name="80nt-Finetune-Specificity-discriminator-on-Manakov2022_train",
        resume_path=RESUME_CKPT,
        start_epoch = START_EPOCH
    )
    print(f"\nDone. Best specificity loss = {best:.4f}")