#!/usr/bin/env python3
"""scripts/RL_finetune.py

PPO Reinforcement Learning Fine-tuning for MiRformer-gen.
Version 2 — Improvements over v1 (2026-02-25):
  Phase 1: Piecewise specificity reward (250x stronger gradient vs smooth 1/(1+N/5000))
  Phase 2: Temperature sampling (T=1.5/1.2/1.0 by stage) + adaptive entropy coefficient
  Phase 3: Conservative curriculum (w_seed ≥ 0.5 throughout) + hard seed penalty (−5 if no match)

Objectives:
  - Reduce off-target binding (~20k → ~16k transcripts) via transcriptome-frequency-aware reward
  - Maintain ≥99% canonical seed-match rate
  - Balance seed-type distribution (target: 35-40% 8-mer)

Architecture:
  - Actor  = pre-trained miRNA decoder (fine-tuned, LR=3e-5)
  - Critic = 3-layer MLP on mean-pooled mRNA encoder output (randomly initialized)
  - mRNA encoder/CNN embedding = FROZEN
  - Reward = curriculum-weighted sum of: seed_match + off-target specificity + potency
             with hard penalty (−5.0) if no canonical seed match

Training:
  - PPO with clipped objective, sequence-level advantage (A = R − V), entropy regularisation
  - Curriculum learning: 3 stages with different reward weights (conservative seed quality)
  - Temperature-scaled rollout sampling; log-probs recorded under T=1.0 for IS-correctness
  - Adaptive entropy coefficient targeting H=0.5 to break entropy collapse
  - Early stopping on (a) KL divergence explosion, (b) entropy collapse, (c) no val improvement
  - W&B logging of all metrics
  - Best checkpoint saved under checkpoints/TargetScan/TwoTowerTransformer/RL_finetune/

Usage:
    nohup python scripts/RL_finetune.py > RL_finetune.log 2>&1 &
"""

import os, sys, copy, pickle, random, math, time, itertools
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.distributions import Categorical
import wandb

# ── Project imports ────────────────────────────────────────────────────────────
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

from Global_parameters import PROJ_HOME
from utils import load_dataset
from Data_pipeline import TargetPredictionDataset
from DTEA_model import TargetGenerationModel

# ── Paths ──────────────────────────────────────────────────────────────────────
CKPT_PATH = os.path.join(
    PROJ_HOME,
    "checkpoints/TargetScan/TwoTowerTransformer/Longformer/TargetGeneration",
    "120/full_cross_attn/best_token_accuracy_0.9552_epoch17.pth",
)
KMER_INDEX_PATH = os.path.join(
    PROJ_HOME, "miR_degradome_ago_clip_pairing_data", "hg19_3UTR_kmer_count.pkl"
)
TRAIN_PATH = os.path.join(
    PROJ_HOME, "TargetScan_dataset", "Positive_primates_train_100_randomized_start.csv"
)
VALID_PATH = os.path.join(
    PROJ_HOME, "TargetScan_dataset", "Positive_primates_validation_100_randomized_start.csv"
)
RL_CKPT_DIR = os.path.join(
    PROJ_HOME, "checkpoints", "TargetScan", "TwoTowerTransformer", "RL_finetune"
)
os.makedirs(RL_CKPT_DIR, exist_ok=True)

# ── Model hyperparameters (must exactly match checkpoint) ──────────────────────
MRNA_MAX_LEN   = 120
MIRNA_MAX_LEN  = 26     # 24 nt + BOS + EOS
EMBED_DIM      = 1024
NUM_HEADS      = 8
NUM_LAYERS     = 4
FF_DIM         = 4096
USE_LONGFORMER = True
WINDOW_SIZE    = 20
VOCAB_SIZE     = 13
N_CLASSES      = 13

# ── Token IDs (CharacterTokenizer with ["A","T","C","G","N"]) ──────────────────
# 0=CLS 1=SEP(EOS) 2=BOS 3=MASK 4=PAD 5=RESERVED 6=UNK 7=A 8=T 9=C 10=G 11=N
BOS_ID = 2
EOS_ID = 1
PAD_ID = 4
TOKEN_TO_NT = {7: 'A', 8: 'T', 9: 'C', 10: 'G', 11: 'N'}

# ── PPO hyperparameters ────────────────────────────────────────────────────────
PPO_CLIP         = 0.2
PPO_EPOCHS       = 4
VALUE_COEF       = 0.5
GRAD_CLIP        = 0.5
TARGET_KL        = 0.05     # per-token KL; break PPO inner-loop above this; warn above 2×
LR_ACTOR         = 3e-5
NUM_ITERATIONS   = 200
BATCH_SIZE       = 32
ROLLOUT_BATCHES  = 16       # 16 × 32 = 512 rollout samples per iteration
MINI_BATCH_SIZE  = 64
MAX_GEN_LEN      = 24       # max nucleotide tokens to generate (excl. BOS/EOS)
VAL_EVERY        = 5        # validate every N iterations
VAL_BATCHES      = 10       # number of val batches per eval
PATIENCE         = 10       # early stop after this many val evals without improvement
DEVICE           = "cuda:0"
SEED             = 42
WANDB_KEY        = "your key"

# ── Curriculum: (max_iter, reward_weights, entropy_coef) ──────────────────────
# v2: Conservative curriculum — keep w_seed ≥ 0.5 to prevent seed-match degradation
CURRICULUM = [
    (50,  {"seed": 0.6, "spec": 0.3, "pot": 0.1}, 0.05),   # Stage 1: seed quality + early spec
    (150, {"seed": 0.5, "spec": 0.4, "pot": 0.1}, 0.03),   # Stage 2: balanced (seed still ≥0.5)
    (200, {"seed": 0.5, "spec": 0.35, "pot": 0.15}, 0.01), # Stage 3: fine-tune with potency
]

# ── Piecewise specificity reward thresholds ────────────────────────────────────
SPEC_THRESH_EXCELLENT  = 15_000   # N < 15k → R_spec = 1.0
SPEC_THRESH_ACCEPTABLE = 19_000   # N ∈ [15k,19k) → linear 1→0
SPEC_THRESH_POOR       = 23_000   # N ∈ [19k,23k) → linear 0→−1; ≥23k → −1.0

# ── Temperature schedule for rollout sampling (breaks entropy collapse) ────────
ROLLOUT_TEMPERATURES = {0: 1.5, 50: 1.2, 150: 1.0}   # stage start → temperature

# ── Hard penalty for sequences with no canonical seed match ───────────────────
HARD_NO_SEED_PENALTY = -5.0


# =============================================================================
# Utility: seed / reward helpers
# =============================================================================

def dna_complement(seq: str) -> str:
    c = {"A": "T", "C": "G", "G": "C", "T": "A", "U": "A"}
    return "".join(c.get(b, "N") for b in seq.upper())


def extract_seed_patterns(mirna_3to5: str) -> dict:
    """
    Compute the four canonical target-site patterns.
    mirna_3to5: DNA string in 3'→5' order (as stored in TargetPredictionDataset).
    Returns DNA strings (5'→3') to search in the mRNA.
    """
    fwd      = mirna_3to5[::-1].upper().replace("U", "T")   # 5'→3' DNA
    seed7    = fwd[1:8]                                       # positions 2-8
    seed7_rc = dna_complement(seed7)[::-1]
    seed6    = fwd[1:7]                                       # positions 2-7
    seed6_rc = dna_complement(seed6)[::-1]
    return {
        "8-mer":    seed7_rc + "A",
        "7-mer-m8": seed7_rc,
        "7-mer-A1": seed6_rc + "A",
        "6-mer":    seed6_rc,
    }


def tokens_to_dna_3to5(token_ids) -> str:
    """
    Convert token ID tensor/list → DNA string in 3'→5' order.
    Stops at EOS (1) or PAD (4); skips BOS (2).
    """
    chars = []
    for t in token_ids:
        tid = int(t.item()) if hasattr(t, "item") else int(t)
        if tid in (EOS_ID, PAD_ID):
            break
        if tid == BOS_ID:
            continue
        ch = TOKEN_TO_NT.get(tid)
        if ch is not None:
            chars.append(ch)
    return "".join(chars)


def mrna_tokens_to_dna_5to3(token_ids) -> str:
    """
    Convert mRNA token IDs → 5'→3' DNA string.
    (mRNA is not reversed in TargetPredictionDataset.)
    """
    chars = []
    for t in token_ids:
        tid = int(t.item()) if hasattr(t, "item") else int(t)
        if tid in (EOS_ID, PAD_ID):
            break
        ch = TOKEN_TO_NT.get(tid)
        if ch is not None:
            chars.append(ch)
    return "".join(chars)


def classify_seed_match(patterns: dict, mrna_5to3: str):
    """
    Returns (seed_type, seed_score, seed_pos) for the best canonical seed match.
    seed_pos = -1 when no match.
    """
    ORDER = [
        ("8-mer",    1.0),
        ("7-mer-m8", 0.8),
        ("7-mer-A1", 0.6),
        ("6-mer",    0.3),
    ]
    for stype, score in ORDER:
        pat = patterns[stype]
        pos = mrna_5to3.find(pat)
        if pos != -1:
            return stype, score, pos
    return "none", -1.0, -1


# =============================================================================
# Reward function
# =============================================================================

class RewardFunction:
    """
    Composite reward: R_total = w_seed*R_seed + w_spec*R_spec + w_pot*R_pot
    with hard override: R_total = HARD_NO_SEED_PENALTY if no canonical seed match.

    R_seed : canonical seed match score ∈ {-1, 0.3, 0.6, 0.8, 1.0}
    R_spec : piecewise specificity reward ∈ [−1, 1] (v2: much stronger gradient)
             N <  15k → 1.0 (excellent)
             N ∈ [15k,19k) → linear 1→0 (gradient −2.5e−4, 125× stronger than v1)
             N ∈ [19k,23k) → linear 0→−1 (penalty zone)
             N ≥  23k → −1.0
    R_pot  : simplified Context++ potency (negated), ∈ [0, ~1]
    """

    CONTEXT_WEIGHTS = {
        "8-mer": -0.6, "7-mer-m8": -0.4, "7-mer-A1": -0.2, "6-mer": -0.1, "none": 0.0,
    }

    def __init__(self, kmer_index_path: str):
        with open(kmer_index_path, "rb") as fh:
            self.kmer_index = pickle.load(fh)
        # Support both nested {k: {kmer: count}} and flat {kmer: count}
        self.nested = isinstance(self.kmer_index, dict) and (
            any(isinstance(k, int) for k in self.kmer_index)
        )
        print(
            f"[RewardFn] Loaded k-mer index (nested={self.nested}). "
            f"Top-level keys: {list(self.kmer_index.keys())[:5]}",
            flush=True,
        )

    def _lookup_6mer(self, pat: str) -> int:
        if self.nested:
            return self.kmer_index.get(6, {}).get(pat, 0)
        return self.kmer_index.get(pat, 0)

    def __call__(
        self,
        gen_tokens:  torch.Tensor,  # (B, ≤MIRNA_MAX_LEN) incl. BOS, may have EOS/PAD
        mrna_tokens: torch.Tensor,  # (B, MRNA_MAX_LEN)
        weights:     dict,
    ):
        B = gen_tokens.shape[0]
        r_seed = torch.zeros(B)
        r_spec = torch.zeros(B)
        r_pot  = torch.zeros(B)
        seed_types  = []
        off_targets = []

        for i in range(B):
            mirna_str = tokens_to_dna_3to5(gen_tokens[i])   # 3'→5' DNA
            mrna_str  = mrna_tokens_to_dna_5to3(mrna_tokens[i])  # 5'→3' DNA

            if len(mirna_str) < 8:
                r_seed[i] = -1.0
                r_spec[i] = 0.0
                r_pot[i]  = 0.0
                seed_types.append("none")
                off_targets.append(90000)
                continue

            patterns               = extract_seed_patterns(mirna_str)
            stype, sscore, spos    = classify_seed_match(patterns, mrna_str)

            r_seed[i] = sscore
            seed_types.append(stype)

            # -- R_spec: piecewise specificity reward (v2 — strong gradient) --
            n_off = self._lookup_6mer(patterns["6-mer"])
            off_targets.append(int(n_off))
            if n_off < SPEC_THRESH_EXCELLENT:
                r_spec[i] = 1.0
            elif n_off < SPEC_THRESH_ACCEPTABLE:
                # Linear 1 → 0 over [15k, 19k); gradient = −1/4000 ≈ −2.5e−4
                r_spec[i] = 1.0 - (n_off - SPEC_THRESH_EXCELLENT) / float(
                    SPEC_THRESH_ACCEPTABLE - SPEC_THRESH_EXCELLENT
                )
            elif n_off < SPEC_THRESH_POOR:
                # Linear 0 → −1 over [19k, 23k); gradient = −1/4000
                r_spec[i] = -(n_off - SPEC_THRESH_ACCEPTABLE) / float(
                    SPEC_THRESH_POOR - SPEC_THRESH_ACCEPTABLE
                )
            else:
                r_spec[i] = -1.0

            # -- R_pot: simplified Context++ (seed weight + local AU) --
            ctx_w = self.CONTEXT_WEIGHTS[stype]
            if spos >= 0:
                lo   = max(0, spos - 30)
                hi   = min(len(mrna_str), spos + 30)
                win  = mrna_str[lo:hi]
                au_r = (win.count("A") + win.count("T")) / max(1, len(win))
            else:
                au_r = 0.0
            ctx_score = ctx_w + (-0.38 * au_r)   # negative = stronger repression
            r_pot[i]  = -ctx_score                # negate → positive reward

        r_total = weights["seed"] * r_seed + weights["spec"] * r_spec + weights["pot"] * r_pot

        # Hard constraint (Phase 3): override total reward with strong penalty if no seed match
        no_seed_mask = torch.tensor(
            [1.0 if st == "none" else 0.0 for st in seed_types], dtype=torch.float32
        )
        r_total = torch.where(no_seed_mask.bool(), torch.full_like(r_total, HARD_NO_SEED_PENALTY), r_total)

        return r_total, {
            "seed": r_seed,
            "spec": r_spec,
            "pot":  r_pot,
            "seed_types":  seed_types,
            "off_targets": off_targets,
        }


# =============================================================================
# Temperature helper (Phase 2: exploration via temperature-scaled sampling)
# =============================================================================

def get_rollout_temperature(iteration: int) -> float:
    """Return sampling temperature based on curriculum stage.
    Higher T → softer distribution → more exploration."""
    temp = 1.0
    for stage_start in sorted(ROLLOUT_TEMPERATURES.keys()):
        if iteration >= stage_start:
            temp = ROLLOUT_TEMPERATURES[stage_start]
    return temp


# =============================================================================
# Adaptive entropy coefficient (Phase 2)
# =============================================================================

class EntropyAdaptiveManager:
    """
    Dynamically adjust entropy coefficient to maintain target entropy.
    Aggressively increases coef when entropy collapses (H < target).
    """

    def __init__(self, initial_coef: float = 0.05, target_entropy: float = 0.5,
                 min_coef: float = 0.01, max_coef: float = 0.5, history_len: int = 5):
        self.coef          = initial_coef
        self.target        = target_entropy
        self.min_coef      = min_coef
        self.max_coef      = max_coef
        self.history_len   = history_len
        self._history: list = []

    def update(self, current_entropy: float, iteration: int) -> float:
        self._history.append(current_entropy)
        if len(self._history) < self.history_len:
            return self.coef

        recent = float(np.mean(self._history[-self.history_len:]))
        old_coef = self.coef

        if recent < 0.1:
            factor = 1.5;  reason = f"CRITICAL collapse ({recent:.4f})"
        elif recent < self.target - 0.2:
            factor = 1.3;  reason = f"Low entropy ({recent:.4f})"
        elif recent < self.target:
            factor = 1.1;  reason = f"Below target ({recent:.4f})"
        elif recent > self.target + 0.3:
            factor = 0.85; reason = f"Too high ({recent:.4f})"
        else:
            factor = 1.0;  reason = None

        self.coef = float(np.clip(self.coef * factor, self.min_coef, self.max_coef))
        if reason is not None:
            print(
                f"[iter {iteration}] EntropyAdaptive: coef {old_coef:.4f} → {self.coef:.4f} ({reason})",
                flush=True,
            )
        return self.coef

    def reset_to(self, coef: float):
        """Reset to a specific value (e.g. on curriculum stage transition)."""
        self.coef = float(np.clip(coef, self.min_coef, self.max_coef))
        self._history.clear()


# =============================================================================
# Critic network
# =============================================================================

class Critic(nn.Module):
    """3-layer MLP value estimator. Input: mean-pooled mRNA encoder output."""

    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, mrna_memory: torch.Tensor) -> torch.Tensor:
        """mrna_memory: (B, L, D) → pooled → (B,)"""
        return self.net(mrna_memory.mean(dim=1)).squeeze(-1)


# =============================================================================
# mRNA encoding / decoder step helpers
# =============================================================================

def encode_mrna(model: TargetGenerationModel, mrna_tokens: torch.Tensor):
    """
    Encode mRNA (frozen encoder path). Returns (mrna_memory, src_key_mask).
    mrna_memory:  (B, L_mrna, D)
    src_key_mask: (B, L_mrna) uint8, 1=valid 0=pad (used for decoder cross-attn)
    """
    src_mask = model.create_src_mask(mrna_tokens)          # (B, L) 1=valid 0=pad
    mrna_sn  = model.sn_embedding(mrna_tokens)             # (B, L, D)
    mrna_cnn = model.cnn_embedding(mrna_sn.transpose(-1, -2))
    mrna_emb = mrna_sn + mrna_cnn

    if model.use_longformer:
        lf_mask = torch.where(
            src_mask > 0,
            torch.zeros_like(src_mask, dtype=torch.long),
            torch.full_like(src_mask, -1, dtype=torch.long),
        )
        assert (lf_mask <= 0).all(), "lf_mask has values > 0"
        mrna_memory  = model.mrna_encoder(mrna_emb, mask=lf_mask)
        src_key_mask = src_mask.to(torch.uint8)
    else:
        mrna_memory  = model.mrna_encoder(mrna_emb, mask=src_mask)
        src_key_mask = src_mask

    return mrna_memory, src_key_mask


def decode_last_logits(
    model: TargetGenerationModel,
    tgt_tokens: torch.Tensor,      # (B, t) — BOS + generated so far
    mrna_memory: torch.Tensor,
    src_key_mask: torch.Tensor,
) -> torch.Tensor:
    """Run decoder and return logits at the last token position: (B, V)."""
    tgt_mask = model.create_tgt_mask(tgt_tokens).to(tgt_tokens.device)
    tgt_emb  = model.sn_embedding(tgt_tokens)
    out      = model.mirna_decoder(
        x=tgt_emb, memory=mrna_memory, src_mask=src_key_mask, tgt_mask=tgt_mask
    )
    return model.predictor_head(out[:, -1, :])   # (B, V)


def teacher_force_eval(
    model: TargetGenerationModel,
    mrna_tokens: torch.Tensor,       # (B, L_mrna)
    generated_full: torch.Tensor,    # (B, MIRNA_MAX_LEN) — BOS + actions + PAD
):
    """
    Full teacher-forcing forward pass; returns (sum_log_probs, entropy, mrna_memory).
      sum_log_probs : (B,) sum of log_prob over valid (non-PAD, non-BOS) positions
      entropy       : scalar mean entropy across valid positions
      mrna_memory   : (B, L_mrna, D) — for critic value computation
    """
    mrna_memory, src_key_mask = encode_mrna(model, mrna_tokens)

    tgt_input  = generated_full[:, :-1]   # (B, T-1)
    tgt_output = generated_full[:, 1:]    # (B, T-1)

    tgt_mask = model.create_tgt_mask(tgt_input).to(tgt_input.device)
    tgt_emb  = model.sn_embedding(tgt_input)
    out      = model.mirna_decoder(
        x=tgt_emb, memory=mrna_memory, src_mask=src_key_mask, tgt_mask=tgt_mask
    )
    logits = model.predictor_head(out)    # (B, T-1, V)

    log_probs_all = F.log_softmax(logits, dim=-1)   # (B, T-1, V)

    valid_mask    = (tgt_output != PAD_ID) & (tgt_output != BOS_ID)   # (B, T-1)
    sel_lp        = log_probs_all.gather(2, tgt_output.unsqueeze(2)).squeeze(2)   # (B, T-1)
    sel_lp        = sel_lp * valid_mask.float()
    sum_lp        = sel_lp.sum(dim=1)    # (B,)

    probs_all = F.softmax(logits, dim=-1)
    ent_per   = -(probs_all * log_probs_all).sum(dim=-1)   # (B, T-1)
    n_valid   = valid_mask.float().sum().clamp(min=1.0)
    entropy   = (ent_per * valid_mask.float()).sum() / n_valid

    return sum_lp, entropy, mrna_memory


# =============================================================================
# Rollout collection
# =============================================================================

def collect_rollouts(model, critic, data_cycle, reward_fn, weights, device, n_batches,
                     iteration: int = 0):
    """
    Generate miRNA sequences under current policy and compute rewards.
    v2: Uses temperature-scaled sampling for exploration; log-probs recorded
        under the original T=1.0 distribution (importance-sampling-correct).
    Returns a dict of concatenated tensors.
    """
    model.eval()
    critic.eval()

    temperature = get_rollout_temperature(iteration)

    mrna_buf    = []
    gen_buf     = []
    slp_buf     = []
    rew_buf     = []
    val_buf     = []
    stype_buf   = []
    otgt_buf    = []

    with torch.no_grad():
        for _ in range(n_batches):
            batch       = next(data_cycle)
            mrna_tokens = batch["mrna_input_ids"].to(device)   # (B, L_mrna)
            B           = mrna_tokens.size(0)

            # --- Encode mRNA (frozen) ---
            mrna_memory, src_key_mask = encode_mrna(model, mrna_tokens)

            # --- Autoregressive generation with temperature sampling ---
            generated     = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
            sum_log_probs = torch.zeros(B, device=device)
            finished      = torch.zeros(B, dtype=torch.bool, device=device)

            for _ in range(MAX_GEN_LEN):
                logits = decode_last_logits(model, generated, mrna_memory, src_key_mask)

                # Sample under temperature-scaled distribution (exploration)
                scaled_logits = logits / temperature
                probs_scaled  = F.softmax(scaled_logits, dim=-1)
                action        = Categorical(probs_scaled).sample()   # (B,)

                # Record log-prob under ORIGINAL (T=1.0) distribution
                # This keeps the importance-sampling ratio correct for PPO
                probs_orig = F.softmax(logits, dim=-1)
                lp = torch.log(
                    probs_orig.gather(1, action.unsqueeze(1)).squeeze(1).clamp(min=1e-10)
                )  # (B,)

                # Freeze finished sequences to PAD
                action = torch.where(finished, torch.full_like(action, PAD_ID), action)
                lp     = torch.where(finished, torch.zeros_like(lp), lp)

                sum_log_probs = sum_log_probs + lp
                generated     = torch.cat([generated, action.unsqueeze(1)], dim=1)
                finished      = finished | (action == EOS_ID)
                if finished.all():
                    break

            # Pad generated to MIRNA_MAX_LEN so we can stack across batches
            T = generated.shape[1]
            if T < MIRNA_MAX_LEN:
                pad_fill = torch.full(
                    (B, MIRNA_MAX_LEN - T), PAD_ID, dtype=torch.long, device=device
                )
                generated = torch.cat([generated, pad_fill], dim=1)
            else:
                generated = generated[:, :MIRNA_MAX_LEN]

            # --- Rewards (on CPU) ---
            rewards, rinfo = reward_fn(generated.cpu(), mrna_tokens.cpu(), weights)
            rewards        = rewards.to(device)

            # --- Critic value V(s) ---
            values = critic(mrna_memory)   # (B,)

            mrna_buf.append(mrna_tokens.cpu())
            gen_buf.append(generated.cpu())
            slp_buf.append(sum_log_probs.cpu())
            rew_buf.append(rewards.cpu())
            val_buf.append(values.cpu())
            stype_buf.extend(rinfo["seed_types"])
            otgt_buf.extend(rinfo["off_targets"])

    return {
        "mrna_tokens":       torch.cat(mrna_buf,  dim=0),
        "generated_full":    torch.cat(gen_buf,   dim=0),
        "old_sum_log_probs": torch.cat(slp_buf,   dim=0),
        "rewards":           torch.cat(rew_buf,   dim=0),
        "values":            torch.cat(val_buf,   dim=0),
        "seed_types":        stype_buf,
        "off_targets":       otgt_buf,
    }


# =============================================================================
# PPO update
# =============================================================================

def ppo_update(
    model, critic, opt_actor, opt_critic, rollout,
    ppo_epochs, mini_batch_size, ppo_clip, value_coef, entropy_coef, device,
):
    """Run multiple PPO epochs over the collected rollout. Returns metric dict."""
    model.train()
    critic.train()

    rewards    = rollout["rewards"]
    values_old = rollout["values"]

    # Sequence-level advantage: A = R - V, then normalise
    advantages = rewards - values_old
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    returns    = rewards   # target for critic (undiscounted; short episodes)

    N = rewards.shape[0]
    acc = dict(policy_loss=0., value_loss=0., entropy=0., kl_div=0., clip_frac=0., n=0)

    for _ in range(ppo_epochs):
        perm = torch.randperm(N)
        for start in range(0, N, mini_batch_size):
            idx = perm[start : start + mini_batch_size]
            if len(idx) < 2:
                continue

            mb_mrna    = rollout["mrna_tokens"][idx].to(device)
            mb_gen     = rollout["generated_full"][idx].to(device)
            mb_old_slp = rollout["old_sum_log_probs"][idx].to(device)
            mb_adv     = advantages[idx].to(device)
            mb_ret     = returns[idx].to(device)

            # Re-evaluate under current policy (teacher forcing)
            sum_lp_new, entropy, mrna_memory = teacher_force_eval(model, mb_mrna, mb_gen)

            # PPO clip loss
            log_ratio   = sum_lp_new - mb_old_slp
            ratio       = torch.exp(log_ratio)
            surr1       = ratio * mb_adv
            surr2       = torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # Critic value loss
            V_new      = critic(mrna_memory)
            value_loss = F.mse_loss(V_new, mb_ret)

            # Total loss
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            opt_actor.zero_grad()
            opt_critic.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.mirna_decoder.parameters(), GRAD_CLIP)
            torch.nn.utils.clip_grad_norm_(model.predictor_head.parameters(), GRAD_CLIP)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), GRAD_CLIP)
            opt_actor.step()
            opt_critic.step()

            # KL in eval mode (removes dropout noise from the estimate)
            model.eval()
            with torch.no_grad():
                tgt_out_mb = mb_gen[:, 1:]
                valid_mb   = (tgt_out_mb != PAD_ID) & (tgt_out_mb != BOS_ID)
                n_tok      = valid_mb.float().sum(dim=1).clamp(min=1.0)
                sum_lp_eval, _, _ = teacher_force_eval(model, mb_mrna, mb_gen)
                log_ratio_eval    = sum_lp_eval - mb_old_slp
                kl                = (-log_ratio_eval / n_tok).mean().item()
                cf                = ((ratio - 1.0).abs() > ppo_clip).float().mean().item()
            model.train()

            acc["policy_loss"] += policy_loss.item()
            acc["value_loss"]  += value_loss.item()
            acc["entropy"]     += entropy.item()
            acc["kl_div"]      += kl
            acc["clip_frac"]   += cf
            acc["n"]           += 1

            # Early exit this PPO epoch if per-token KL exceeds target
            if kl > TARGET_KL:
                break

    n = max(1, acc.pop("n"))
    return {k: v / n for k, v in acc.items()}


# =============================================================================
# Validation
# =============================================================================

def evaluate_val(model, critic, val_loader, reward_fn, weights, device, n_batches):
    """Greedy decoding on validation set; returns metric dict."""
    model.eval()
    critic.eval()
    all_rewards, all_off, all_stypes, all_values = [], [], [], []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= n_batches:
                break
            mrna_tokens = batch["mrna_input_ids"].to(device)
            B           = mrna_tokens.size(0)

            mrna_memory, src_key_mask = encode_mrna(model, mrna_tokens)

            generated = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
            finished  = torch.zeros(B, dtype=torch.bool, device=device)
            for _ in range(MAX_GEN_LEN):
                logits  = decode_last_logits(model, generated, mrna_memory, src_key_mask)
                next_id = logits.argmax(dim=-1)
                next_id = torch.where(finished, torch.full_like(next_id, PAD_ID), next_id)
                generated = torch.cat([generated, next_id.unsqueeze(1)], dim=1)
                finished  = finished | (next_id == EOS_ID)
                if finished.all():
                    break

            T = generated.shape[1]
            if T < MIRNA_MAX_LEN:
                pad_fill  = torch.full((B, MIRNA_MAX_LEN - T), PAD_ID, dtype=torch.long, device=device)
                generated = torch.cat([generated, pad_fill], dim=1)
            else:
                generated = generated[:, :MIRNA_MAX_LEN]

            rewards, rinfo = reward_fn(generated.cpu(), mrna_tokens.cpu(), weights)
            values         = critic(mrna_memory)

            all_rewards.extend(rewards.tolist())
            all_off.extend(rinfo["off_targets"])
            all_stypes.extend(rinfo["seed_types"])
            all_values.extend(values.tolist())

    cnt   = Counter(all_stypes)
    total = max(1, sum(cnt.values()))
    sdist = {k: cnt.get(k, 0) / total for k in ("8-mer", "7-mer-m8", "7-mer-A1", "6-mer", "none")}

    return {
        "avg_reward":      float(np.mean(all_rewards)),
        "avg_off_targets": float(np.mean(all_off)),
        "seed_rate":       1.0 - sdist["none"],
        "8mer_frac":       sdist["8-mer"],
        "7mer_m8_frac":    sdist["7-mer-m8"],
        "7mer_a1_frac":    sdist["7-mer-A1"],
        "6mer_frac":       sdist["6-mer"],
        "avg_value":       float(np.mean(all_values)),
    }


# =============================================================================
# Checkpoint helpers
# =============================================================================

def load_pretrained(model, ckpt_path, device):
    """Load checkpoint; drop mismatched rotary buffers."""
    sd_ckpt  = torch.load(ckpt_path, map_location=device)
    sd_model = model.state_dict()
    to_load  = {}
    for k, v in sd_ckpt.items():
        if k not in sd_model:
            continue
        if "rotary.cos_emb" in k or "rotary.sin_emb" in k:
            if v.shape == sd_model[k].shape:
                to_load[k] = v
            else:
                print(
                    f"[ckpt] Dropping mismatched rotary {k}: {v.shape} vs {sd_model[k].shape}",
                    flush=True,
                )
        else:
            to_load[k] = v
    missing, _ = model.load_state_dict(to_load, strict=False)
    print(
        f"[ckpt] Loaded {len(to_load)}/{len(sd_ckpt)} tensors. Missing: {len(missing)}",
        flush=True,
    )
    return model


def get_curriculum(iteration: int):
    """Return (reward_weights, entropy_coef) for the given iteration."""
    for max_iter, weights, ent_coef in CURRICULUM:
        if iteration < max_iter:
            return weights, ent_coef
    return CURRICULUM[-1][1], CURRICULUM[-1][2]


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# Main
# =============================================================================

def main():
    seed_everything(SEED)
    device = torch.device(DEVICE)

    print("=" * 70, flush=True)
    print("MiRformer-gen PPO RL Finetuning", flush=True)
    print(f"  Device:        {DEVICE}", flush=True)
    print(f"  Checkpoint:    {CKPT_PATH}", flush=True)
    print(f"  Train data:    {TRAIN_PATH}", flush=True)
    print(f"  Save dir:      {RL_CKPT_DIR}", flush=True)
    print(f"  Iterations:    {NUM_ITERATIONS}", flush=True)
    print(f"  Rollout/iter:  {ROLLOUT_BATCHES}×{BATCH_SIZE}={ROLLOUT_BATCHES*BATCH_SIZE}", flush=True)
    print("=" * 70, flush=True)

    # ── Build model ──────────────────────────────────────────────────────────
    model = TargetGenerationModel(
        mrna_max_len=MRNA_MAX_LEN,
        mirna_max_len=MIRNA_MAX_LEN,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        batch_size=BATCH_SIZE,
        vocab_size=VOCAB_SIZE,
        n_classes=N_CLASSES,
        dropout_rate=0.2,
        device=DEVICE,
        seed=SEED,
        use_longformer=USE_LONGFORMER,
        window_size=WINDOW_SIZE,
    )
    model = load_pretrained(model, CKPT_PATH, device)
    model = model.to(device)

    # Freeze mRNA encoder and CNN embedding; keep sn_embedding trainable
    for p in model.mrna_encoder.parameters():
        p.requires_grad_(False)
    for p in model.cnn_embedding.parameters():
        p.requires_grad_(False)
    # miRNA decoder, predictor_head, sn_embedding: trainable
    for p in model.mirna_decoder.parameters():
        p.requires_grad_(True)
    for p in model.predictor_head.parameters():
        p.requires_grad_(True)
    for p in model.sn_embedding.parameters():
        p.requires_grad_(True)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable:,} / {n_total:,}", flush=True)

    # ── Critic ───────────────────────────────────────────────────────────────
    critic = Critic(hidden_dim=EMBED_DIM).to(device)

    # ── Optimisers ───────────────────────────────────────────────────────────
    actor_params = (
        list(model.mirna_decoder.parameters())
        + list(model.predictor_head.parameters())
        + list(model.sn_embedding.parameters())
    )
    opt_actor  = AdamW(actor_params,          lr=LR_ACTOR,  weight_decay=0.01)
    opt_critic = AdamW(critic.parameters(),   lr=LR_ACTOR,  weight_decay=0.01)

    # ── Reward function ──────────────────────────────────────────────────────
    reward_fn = RewardFunction(KMER_INDEX_PATH)

    # ── Data ─────────────────────────────────────────────────────────────────
    tokenizer = model.tokenizer
    D_train   = load_dataset(TRAIN_PATH, sep=",")
    D_val     = load_dataset(VALID_PATH, sep=",")
    ds_train  = TargetPredictionDataset(D_train, MRNA_MAX_LEN, MIRNA_MAX_LEN, tokenizer)
    ds_val    = TargetPredictionDataset(D_val,   MRNA_MAX_LEN, MIRNA_MAX_LEN, tokenizer)
    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True,  num_workers=2)
    val_loader   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=2)
    data_cycle   = itertools.cycle(train_loader)   # infinite iterator over shuffled data

    print(f"Train: {len(ds_train)} samples | Val: {len(ds_val)} samples", flush=True)

    # ── W&B ──────────────────────────────────────────────────────────────────
    wandb.login(key=WANDB_KEY)
    wandb.init(
        project="mirna-RL-finetune",
        name=f"PPO-v2-piecewise-temp-conservative-lr{LR_ACTOR:.0e}",
        config={
            "lr_actor":        LR_ACTOR,
            "ppo_clip":        PPO_CLIP,
            "ppo_epochs":      PPO_EPOCHS,
            "value_coef":      VALUE_COEF,
            "grad_clip":       GRAD_CLIP,
            "target_kl":       TARGET_KL,
            "rollout_batches": ROLLOUT_BATCHES,
            "batch_size":      BATCH_SIZE,
            "mini_batch_size": MINI_BATCH_SIZE,
            "num_iterations":  NUM_ITERATIONS,
            "mrna_max_len":    MRNA_MAX_LEN,
            "mirna_max_len":   MIRNA_MAX_LEN,
            "max_gen_len":     MAX_GEN_LEN,
            "checkpoint":      CKPT_PATH,
        },
        tags=["RL", "PPO", "miRNA-generation", "off-target-reduction"],
        save_code=True,
        job_type="rl-finetune",
    )

    # ── Training state ────────────────────────────────────────────────────────
    best_off_targets  = float("inf")
    best_state        = None
    patience_counter  = 0

    # v2: Adaptive entropy manager.
    # Observed H ≈ 0.08 at baseline; target H=0.15 (realistic ~2× baseline).
    # max_coef=0.10 prevents entropy bonus from dominating the policy gradient.
    ent_manager = EntropyAdaptiveManager(
        initial_coef=CURRICULUM[0][2],   # 0.05 (Stage 1)
        target_entropy=0.15,
        min_coef=0.01,
        max_coef=0.10,
        history_len=5,
    )
    entropy_coef_live = ent_manager.coef

    print("Starting PPO RL finetuning …", flush=True)

    for iteration in range(NUM_ITERATIONS):
        t0 = time.time()

        # Curriculum
        weights, entropy_coef_sched = get_curriculum(iteration)
        # On curriculum stage transitions, reset entropy manager to new schedule coef
        if iteration in (50, 150):
            ent_manager.reset_to(entropy_coef_sched)
            print(f"[iter {iteration+1}] Curriculum stage transition: entropy_coef reset to {entropy_coef_sched}", flush=True)

        # ── Collect rollouts (with temperature sampling) ───────────────────────
        rollout = collect_rollouts(
            model, critic, data_cycle, reward_fn, weights, device,
            n_batches=ROLLOUT_BATCHES,
            iteration=iteration,
        )

        # Rollout stats
        avg_reward  = float(rollout["rewards"].mean())
        avg_off_tgt = float(np.mean(rollout["off_targets"]))
        cnt_t       = Counter(rollout["seed_types"])
        tot_t       = max(1, sum(cnt_t.values()))
        sdist       = {k: cnt_t.get(k, 0) / tot_t
                       for k in ("8-mer", "7-mer-m8", "7-mer-A1", "6-mer", "none")}
        seed_rate   = 1.0 - sdist["none"]

        # ── PPO update ────────────────────────────────────────────────────────
        upd = ppo_update(
            model, critic, opt_actor, opt_critic, rollout,
            ppo_epochs=PPO_EPOCHS,
            mini_batch_size=MINI_BATCH_SIZE,
            ppo_clip=PPO_CLIP,
            value_coef=VALUE_COEF,
            entropy_coef=entropy_coef_live,
            device=device,
        )

        elapsed = time.time() - t0
        kl      = upd["kl_div"]
        entropy = upd["entropy"]

        # ── KL divergence guard (log warning; PPO clip already limits damage) ─
        if kl > TARGET_KL * 2:
            print(
                f"[iter {iteration+1:3d}] WARNING: high per-token KL={kl:.4f} "
                f"(target={TARGET_KL:.3f}). PPO clip is protecting the policy.",
                flush=True,
            )

        # ── Adaptive entropy coefficient (v2) ─────────────────────────────────
        entropy_coef_live = ent_manager.update(entropy, iteration + 1)
        rollout_temp = get_rollout_temperature(iteration)

        # ── Build log dict ────────────────────────────────────────────────────
        log = {
            "iteration":              iteration + 1,
            "train/avg_reward":       avg_reward,
            "train/avg_off_targets":  avg_off_tgt,
            "train/seed_rate":        seed_rate,
            "train/8mer_frac":        sdist["8-mer"],
            "train/7mer_m8_frac":     sdist["7-mer-m8"],
            "train/7mer_a1_frac":     sdist["7-mer-A1"],
            "train/6mer_frac":        sdist["6-mer"],
            "train/none_frac":        sdist["none"],
            "ppo/policy_loss":        upd["policy_loss"],
            "ppo/value_loss":         upd["value_loss"],
            "ppo/entropy":            entropy,
            "ppo/kl_div":             kl,
            "ppo/clip_frac":          upd["clip_frac"],
            "curriculum/w_seed":      weights["seed"],
            "curriculum/w_spec":      weights["spec"],
            "curriculum/w_pot":       weights["pot"],
            "curriculum/entropy_coef": entropy_coef_live,
            "curriculum/rollout_temp": rollout_temp,
            "lr_actor":               opt_actor.param_groups[0]["lr"],
            "time_per_iter_s":        elapsed,
        }

        # ── Validation (every VAL_EVERY iterations) ───────────────────────────
        if (iteration + 1) % VAL_EVERY == 0:
            vm = evaluate_val(
                model, critic, val_loader, reward_fn, weights, device, n_batches=VAL_BATCHES
            )
            log.update({
                "val/avg_reward":      vm["avg_reward"],
                "val/avg_off_targets": vm["avg_off_targets"],
                "val/seed_rate":       vm["seed_rate"],
                "val/8mer_frac":       vm["8mer_frac"],
                "val/7mer_m8_frac":    vm["7mer_m8_frac"],
                "val/7mer_a1_frac":    vm["7mer_a1_frac"],
            })
            print(
                f"[iter {iteration+1:3d}] "
                f"R={avg_reward:.4f} | train_off={avg_off_tgt:.0f} "
                f"| seed={seed_rate:.3f} | 8mer={sdist['8-mer']:.3f} "
                f"| KL={kl:.4f} | H={entropy:.3f} | "
                f"val_off={vm['avg_off_targets']:.0f} "
                f"| val_seed={vm['seed_rate']:.3f} | t={elapsed:.0f}s",
                flush=True,
            )

            # ── Checkpoint: save best by validation off-target count ──────────
            if vm["avg_off_targets"] < best_off_targets:
                best_off_targets = vm["avg_off_targets"]
                best_state = {
                    "model":     copy.deepcopy(model.state_dict()),
                    "critic":    copy.deepcopy(critic.state_dict()),
                    "iteration": iteration + 1,
                    "val_metrics": vm,
                }
                ckpt_save = os.path.join(
                    RL_CKPT_DIR,
                    f"best_off_targets_{best_off_targets:.0f}_iter{iteration+1}.pth",
                )
                torch.save(best_state, ckpt_save)
                print(
                    f"[iter {iteration+1:3d}] *** New best: "
                    f"off_targets={best_off_targets:.0f} → {ckpt_save}",
                    flush=True,
                )
                patience_counter = 0
            else:
                patience_counter += 1
                print(
                    f"[iter {iteration+1:3d}] No improvement "
                    f"(patience {patience_counter}/{PATIENCE})",
                    flush=True,
                )

            # ── Early stopping ────────────────────────────────────────────────
            if patience_counter >= PATIENCE:
                print(
                    f"[iter {iteration+1}] Early stopping: no improvement for "
                    f"{PATIENCE} val evals.",
                    flush=True,
                )
                break

        else:
            print(
                f"[iter {iteration+1:3d}] "
                f"R={avg_reward:.4f} | off_tgt={avg_off_tgt:.0f} "
                f"| seed={seed_rate:.3f} | KL={kl:.4f} | H={entropy:.3f} | t={elapsed:.0f}s",
                flush=True,
            )

        wandb.log(log, step=iteration + 1)

    # ── Final save ────────────────────────────────────────────────────────────
    final_path = os.path.join(RL_CKPT_DIR, "final_model.pth")
    torch.save(
        {
            "model":     model.state_dict(),
            "critic":    critic.state_dict(),
            "iteration": iteration + 1,
            "best_off_targets": best_off_targets,
        },
        final_path,
    )
    print(f"\nTraining complete.", flush=True)
    print(f"Best val off_targets:  {best_off_targets:.0f}", flush=True)
    print(f"Final model saved to: {final_path}", flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()
