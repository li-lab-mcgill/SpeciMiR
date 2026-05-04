#!/usr/bin/env python3
"""scripts/RL_finetune_v2.py

PPO Reinforcement Learning Fine-tuning for MiRformer-gen — Version 3.

Addresses all 7 issues from RL_finetune_v2_analysis.md:

  Issue 1 (Rank-based reward): Rank-based r_spec computed across the FULL
           512-sample rollout buffer (was per-batch of 32 — too local).
           Blended 50/50 with piecewise reward.
  Issue 2 (sn_embedding drift): sn_embedding FROZEN — shared with frozen mRNA
           encoder; drift during RL would silently corrupt cross-attention keys.
  Issue 3 (Blind critic): Reverted to mRNA-only Critic (3-layer MLP). The
           CriticWithSeed variant increased value_loss (2.1+) and destabilised
           advantage estimation — mRNA-only critic converges faster and yields
           lower val_off in practice (18,137 vs 19,063).
  Issue 4 (IS-correct log-probs): Log-probs stored from the SAMPLING (tempered)
           distribution π_T, not T=1.0 — correct PPO importance-sampling ratio.
  Issue 5 (Too few val batches): VAL_BATCHES=100 (3,200 samples vs 320).
  Issue 6 (Weak entropy guard): target_entropy=0.15, max_coef=0.25 (calibrated
           to actual model entropy ~0.08; max was 0.5 — too aggressive).
  Issue 7 (Token masking): EOS(1) + ATCG(7-10) allowed; all other tokens masked
           with -1e9 during rollout. Bug fix: prior version excluded EOS from
           NUCLEOTIDE_IDS so sequences could never self-terminate.

Usage:
    nohup python scripts/RL_finetune_v2.py > RL_finetune_v2.log 2>&1 &
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
    PROJ_HOME, "checkpoints", "TargetScan", "TwoTowerTransformer", "RL_finetune_v2_mrna_critic"
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

# Issue 7 fix: allow EOS(1) + ATCG(7-10); mask everything else to -1e9
# Previous version excluded EOS so sequences could never self-terminate (bug!)
VALID_GEN_TOKENS = [1, 7, 8, 9, 10]   # EOS, A, T, C, G

# ── PPO hyperparameters ────────────────────────────────────────────────────────
PPO_CLIP         = 0.2
PPO_EPOCHS       = 4
VALUE_COEF       = 0.5
GRAD_CLIP        = 0.5
TARGET_KL        = 0.05
LR_ACTOR         = 3e-5
NUM_ITERATIONS   = 200
BATCH_SIZE       = 32
ROLLOUT_BATCHES  = 16       # 16 × 32 = 512 rollout samples per iteration
MINI_BATCH_SIZE  = 64
MAX_GEN_LEN      = 24
VAL_EVERY        = 5
VAL_BATCHES      = 100      # Issue 5: 100 × 32 = 3,200 samples
PATIENCE         = 15
DEVICE           = "cuda:0"
SEED             = 42
WANDB_KEY        = "your key"

# ── Reward thresholds (piecewise, Issue 1) ─────────────────────────────────────
SPEC_THRESH_EXCELLENT  = 15_000
SPEC_THRESH_ACCEPTABLE = 18_000
SPEC_THRESH_POOR       = 22_000
INVALID_SEED_PENALTY   = -5.0

# ── Issue 1: rank-based blending across full 512-sample rollout buffer ─────────
RANK_SPEC_WEIGHT = 0.5    # 50% piecewise, 50% rank-based

# ── Temperature schedule (Issue 4 / Issue 7) ──────────────────────────────────
TEMPERATURE_SCHEDULE = {0: 1.4, 50: 1.2, 150: 1.0}

# ── Curriculum (conservative, w_seed ≥ 0.5 always) ───────────────────────────
CURRICULUM = [
    (50,  {"seed": 0.6, "spec": 0.3, "pot": 0.1}, 0.03),
    (150, {"seed": 0.5, "spec": 0.4, "pot": 0.1}, 0.01),
    (200, {"seed": 0.5, "spec": 0.35, "pot": 0.15}, 0.005),
]


# =============================================================================
# Utility: seed / reward helpers
# =============================================================================

def dna_complement(seq: str) -> str:
    c = {"A": "T", "C": "G", "G": "C", "T": "A", "U": "A", "N": "N"}
    return "".join(c.get(b, "N") for b in seq.upper())


def extract_seed_patterns(mirna_3to5: str) -> dict:
    fwd      = mirna_3to5[::-1].upper().replace("U", "T")
    seed7    = fwd[1:8]
    seed7_rc = dna_complement(seed7)[::-1]
    seed6    = fwd[1:7]
    seed6_rc = dna_complement(seed6)[::-1]
    return {
        "8-mer":    seed7_rc + "A",
        "7-mer-m8": seed7_rc,
        "7-mer-A1": seed6_rc + "A",
        "6-mer":    seed6_rc,
    }


def tokens_to_dna_3to5(token_ids) -> str:
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
    ORDER = [
        ("8-mer",    1.0),
        ("7-mer-m8", 0.8),
        ("7-mer-A1", 0.6),
        ("6-mer",    0.3),
    ]
    for stype, score in ORDER:
        pos = mrna_5to3.find(patterns[stype])
        if pos != -1:
            return stype, score, pos
    return "none", -1.0, -1


# =============================================================================
# Reward function — returns per-sample components (no r_total combination)
# r_total is assembled in collect_rollouts after full-buffer rank-based blending
# =============================================================================

class RewardFunction:
    """
    Returns individual reward components; caller combines them.

    R_seed : canonical seed match score ∈ {−1, 0.3, 0.6, 0.8, 1.0}
    R_spec : piecewise specificity reward ∈ [−1, 1]
             N <  15k → 1.0
             N ∈ [15k, 18k) → linear 1→0 (gradient −3.3e−4)
             N ∈ [18k, 22k) → linear 0→−0.5 (gradient −1.25e−4)
             N ≥  22k → −0.5
    R_pot  : simplified Context++ potency (negated) ∈ [0, ~1]
    """

    CONTEXT_WEIGHTS = {
        "8-mer": -0.6, "7-mer-m8": -0.4, "7-mer-A1": -0.2, "6-mer": -0.1, "none": 0.0,
    }

    def __init__(self, kmer_index_path: str):
        with open(kmer_index_path, "rb") as fh:
            self.kmer_index = pickle.load(fh)
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

    def _piecewise_spec(self, n_off: int) -> float:
        if n_off < SPEC_THRESH_EXCELLENT:
            return 1.0
        elif n_off < SPEC_THRESH_ACCEPTABLE:
            return 1.0 - (n_off - SPEC_THRESH_EXCELLENT) / float(
                SPEC_THRESH_ACCEPTABLE - SPEC_THRESH_EXCELLENT)
        elif n_off < SPEC_THRESH_POOR:
            return -0.5 * (n_off - SPEC_THRESH_ACCEPTABLE) / float(
                SPEC_THRESH_POOR - SPEC_THRESH_ACCEPTABLE)
        else:
            return -0.5

    def __call__(self, gen_tokens: torch.Tensor, mrna_tokens: torch.Tensor):
        """
        Returns (r_seed, r_spec_pw, r_pot, seed_types, off_targets, no_seed_mask).
        r_spec_pw is piecewise only — rank-based blending is done in collect_rollouts.
        """
        B = gen_tokens.shape[0]
        r_seed    = torch.zeros(B)
        r_spec_pw = torch.zeros(B)
        r_pot     = torch.zeros(B)
        seed_types  = []
        off_targets = []

        for i in range(B):
            mirna_str = tokens_to_dna_3to5(gen_tokens[i])
            mrna_str  = mrna_tokens_to_dna_5to3(mrna_tokens[i])

            if len(mirna_str) < 8:
                r_seed[i]    = -1.0
                r_spec_pw[i] = -0.5
                r_pot[i]     = 0.0
                seed_types.append("none")
                off_targets.append(90_000)
                continue

            patterns            = extract_seed_patterns(mirna_str)
            stype, sscore, spos = classify_seed_match(patterns, mrna_str)

            r_seed[i] = sscore
            seed_types.append(stype)

            n_off = self._lookup_6mer(patterns["6-mer"])
            off_targets.append(int(n_off))
            r_spec_pw[i] = self._piecewise_spec(n_off)

            ctx_w = self.CONTEXT_WEIGHTS[stype]
            if spos >= 0:
                lo   = max(0, spos - 30)
                hi   = min(len(mrna_str), spos + 30)
                win  = mrna_str[lo:hi]
                au_r = (win.count("A") + win.count("T")) / max(1, len(win))
            else:
                au_r = 0.0
            r_pot[i] = -(ctx_w + (-0.38 * au_r))

        no_seed_mask = torch.tensor(
            [1.0 if st == "none" else 0.0 for st in seed_types], dtype=torch.float32
        )
        return r_seed, r_spec_pw, r_pot, seed_types, off_targets, no_seed_mask


def combine_rewards(r_seed, r_spec_blended, r_pot, no_seed_mask, weights):
    """Combine reward components into r_total with hard no-seed override."""
    r_total = (
        weights["seed"] * r_seed
        + weights["spec"] * r_spec_blended
        + weights["pot"]  * r_pot
    )
    return torch.where(
        no_seed_mask.bool(),
        torch.full_like(r_total, INVALID_SEED_PENALTY),
        r_total,
    )


# =============================================================================
# Adaptive entropy coefficient — Issue 6: calibrated to actual model entropy
# =============================================================================

class EntropyAdaptiveManager:
    """Adjust entropy coefficient to maintain target entropy (target=0.15)."""

    def __init__(self, initial_coef=0.05, target_entropy=0.15,
                 min_coef=0.01, max_coef=0.25, history_len=5):
        self.coef        = initial_coef
        self.target      = target_entropy   # Issue 6: was 0.3, calibrated to 0.15
        self.min_coef    = min_coef
        self.max_coef    = max_coef         # Issue 6: was 0.5, set to 0.25
        self.history_len = history_len
        self.history     = []

    def update(self, current_entropy, iteration):
        self.history.append(current_entropy)
        if len(self.history) < self.history_len:
            return self.coef

        recent   = float(np.mean(self.history[-self.history_len:]))
        old_coef = self.coef

        if recent < 0.05:
            factor = 2.0;  reason = f"CRITICAL collapse ({recent:.4f})"
        elif recent < self.target * 0.5:
            factor = 1.5;  reason = f"Low entropy ({recent:.4f})"
        elif recent < self.target:
            factor = 1.2;  reason = f"Below target ({recent:.4f})"
        elif recent > self.target * 2.5:
            factor = 0.7;  reason = f"Too high ({recent:.4f})"
        else:
            factor = 1.0;  reason = None

        self.coef = float(np.clip(self.coef * factor, self.min_coef, self.max_coef))
        if reason is not None:
            print(
                f"  [EntropyMgr iter {iteration}] coef: {old_coef:.4f} → {self.coef:.4f} "
                f"({reason}, target={self.target:.2f})",
                flush=True,
            )
        return self.coef

    def reset_to(self, coef: float):
        self.coef = float(np.clip(coef, self.min_coef, self.max_coef))
        self.history.clear()


# =============================================================================
# Critic — mRNA-only value estimator (mean-pooled encoder output)
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

def encode_mrna(model, mrna_tokens):
    src_mask = model.create_src_mask(mrna_tokens)
    mrna_sn  = model.sn_embedding(mrna_tokens)
    mrna_cnn = model.cnn_embedding(mrna_sn.transpose(-1, -2))
    mrna_emb = mrna_sn + mrna_cnn

    if model.use_longformer:
        lf_mask = torch.where(
            src_mask > 0,
            torch.zeros_like(src_mask, dtype=torch.long),
            torch.full_like(src_mask, -1, dtype=torch.long),
        )
        assert (lf_mask <= 0).all()
        mrna_memory  = model.mrna_encoder(mrna_emb, mask=lf_mask)
        src_key_mask = src_mask.to(torch.uint8)
    else:
        mrna_memory  = model.mrna_encoder(mrna_emb, mask=src_mask)
        src_key_mask = src_mask
    return mrna_memory, src_key_mask


def decode_last_logits(model, tgt_tokens, mrna_memory, src_key_mask):
    tgt_mask = model.create_tgt_mask(tgt_tokens).to(tgt_tokens.device)
    tgt_emb  = model.sn_embedding(tgt_tokens)
    out      = model.mirna_decoder(
        x=tgt_emb, memory=mrna_memory, src_mask=src_key_mask, tgt_mask=tgt_mask
    )
    return model.predictor_head(out[:, -1, :])


def teacher_force_eval(model, mrna_tokens, generated_full):
    """Teacher-forcing forward pass. Returns (sum_log_probs, entropy, mrna_memory)."""
    mrna_memory, src_key_mask = encode_mrna(model, mrna_tokens)

    tgt_input  = generated_full[:, :-1]
    tgt_output = generated_full[:, 1:]

    tgt_mask = model.create_tgt_mask(tgt_input).to(tgt_input.device)
    tgt_emb  = model.sn_embedding(tgt_input)
    out      = model.mirna_decoder(
        x=tgt_emb, memory=mrna_memory, src_mask=src_key_mask, tgt_mask=tgt_mask
    )
    logits = model.predictor_head(out)

    log_probs_all = F.log_softmax(logits, dim=-1)
    valid_mask    = (tgt_output != PAD_ID) & (tgt_output != BOS_ID)
    sel_lp        = log_probs_all.gather(2, tgt_output.unsqueeze(2)).squeeze(2)
    sel_lp        = sel_lp * valid_mask.float()
    sum_lp        = sel_lp.sum(dim=1)

    probs_all = F.softmax(logits, dim=-1)
    ent_per   = -(probs_all * log_probs_all).sum(dim=-1)
    n_valid   = valid_mask.float().sum().clamp(min=1.0)
    entropy   = (ent_per * valid_mask.float()).sum() / n_valid

    return sum_lp, entropy, mrna_memory


# =============================================================================
# Rollout collection — Issues 1, 3, 4, 7 addressed
# =============================================================================

def get_temperature(iteration: int) -> float:
    temp = 1.0
    for stage_start in sorted(TEMPERATURE_SCHEDULE.keys()):
        if iteration >= stage_start:
            temp = TEMPERATURE_SCHEDULE[stage_start]
    return temp


def build_generation_mask(vocab_size: int, device) -> torch.Tensor:
    """
    Additive logit mask: 0 for valid generation tokens (EOS + ATCG), −1e9 for rest.
    Issue 7 fix: includes EOS(1) so sequences can self-terminate.
    """
    mask = torch.full((vocab_size,), -1e9, device=device)
    for tid in VALID_GEN_TOKENS:
        mask[tid] = 0.0
    return mask


def collect_rollouts(model, critic, data_cycle, reward_fn, weights, device,
                     n_batches, iteration):
    """
    Generate miRNA sequences and collect rewards.

    Issue 1: r_spec rank-based computed across FULL 512-sample buffer after
             all batches (not per-batch of 32).
    Issue 4: log-probs stored from tempered sampling distribution.
    Issue 7: non-nucleotide tokens masked; EOS included.
    """
    model.eval()
    critic.eval()

    temperature = get_temperature(iteration)
    gen_mask    = build_generation_mask(VOCAB_SIZE, device)   # additive

    mrna_buf     = []
    gen_buf      = []
    slp_buf      = []
    rseed_buf    = []    # Issue 1: buffer individual components
    rspec_pw_buf = []    # piecewise r_spec (no rank yet)
    rpot_buf     = []
    noseed_buf   = []    # no-seed mask per sample
    val_buf      = []
    stype_buf    = []
    otgt_buf     = []

    with torch.no_grad():
        for _ in range(n_batches):
            batch       = next(data_cycle)
            mrna_tokens = batch["mrna_input_ids"].to(device)
            B           = mrna_tokens.size(0)

            mrna_memory, src_key_mask = encode_mrna(model, mrna_tokens)

            generated     = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
            sum_log_probs = torch.zeros(B, device=device)
            finished      = torch.zeros(B, dtype=torch.bool, device=device)

            for _ in range(MAX_GEN_LEN):
                logits = decode_last_logits(model, generated, mrna_memory, src_key_mask)

                # Issue 7: apply generation mask (allows EOS + ATCG only)
                logits_masked = logits + gen_mask.unsqueeze(0)

                # Issue 4: sample from tempered distribution
                # old_log_prob = log π_T(a|s) — behavior policy, correct for PPO ratio
                scaled_logits = logits_masked / temperature
                probs         = F.softmax(scaled_logits, dim=-1)
                dist          = Categorical(probs)
                action        = dist.sample()
                lp            = dist.log_prob(action)   # log π_T

                action = torch.where(finished, torch.full_like(action, PAD_ID), action)
                lp     = torch.where(finished, torch.zeros_like(lp), lp)

                sum_log_probs = sum_log_probs + lp
                generated     = torch.cat([generated, action.unsqueeze(1)], dim=1)
                finished      = finished | (action == EOS_ID)
                if finished.all():
                    break

            T = generated.shape[1]
            if T < MIRNA_MAX_LEN:
                pad_fill  = torch.full((B, MIRNA_MAX_LEN - T), PAD_ID, dtype=torch.long, device=device)
                generated = torch.cat([generated, pad_fill], dim=1)
            else:
                generated = generated[:, :MIRNA_MAX_LEN]

            # Reward components (no r_total yet)
            r_seed, r_spec_pw, r_pot, seed_types, off_targets, no_seed_mask = reward_fn(
                generated.cpu(), mrna_tokens.cpu()
            )

            values = critic(mrna_memory)   # (B,)

            mrna_buf.append(mrna_tokens.cpu())
            gen_buf.append(generated.cpu())
            slp_buf.append(sum_log_probs.cpu())
            rseed_buf.append(r_seed)
            rspec_pw_buf.append(r_spec_pw)
            rpot_buf.append(r_pot)
            noseed_buf.append(no_seed_mask)
            val_buf.append(values.cpu())
            stype_buf.extend(seed_types)
            otgt_buf.extend(off_targets)

    # Issue 1: rank-based r_spec across FULL 512-sample buffer
    r_seed_all    = torch.cat(rseed_buf)
    r_spec_pw_all = torch.cat(rspec_pw_buf)
    r_pot_all     = torch.cat(rpot_buf)
    no_seed_all   = torch.cat(noseed_buf)
    off_tensor    = torch.tensor(otgt_buf, dtype=torch.float)

    N             = len(off_tensor)
    ranks         = off_tensor.argsort().argsort().float()          # 0 = best
    r_spec_rank   = 1.0 - 2.0 * ranks / max(1.0, float(N - 1))    # ∈ [-1, +1]
    r_spec_blend  = (1.0 - RANK_SPEC_WEIGHT) * r_spec_pw_all + RANK_SPEC_WEIGHT * r_spec_rank

    rewards = combine_rewards(r_seed_all, r_spec_blend, r_pot_all, no_seed_all, weights)

    invalid_rate = no_seed_all.mean().item()
    print(
        f"  [Rollout iter] T={get_temperature(iteration):.2f} | invalid_seed={invalid_rate:.3f} "
        f"| mean_off={float(off_tensor.mean()):.0f}",
        flush=True,
    )

    return {
        "mrna_tokens":       torch.cat(mrna_buf,  dim=0),
        "generated_full":    torch.cat(gen_buf,   dim=0),
        "old_sum_log_probs": torch.cat(slp_buf,   dim=0),
        "rewards":           rewards,
        "values":            torch.cat(val_buf,   dim=0),
        "seed_types":        stype_buf,
        "off_targets":       otgt_buf,
    }


# =============================================================================
# PPO update — Issue 3: critic receives generated tokens
# =============================================================================

def ppo_update(
    model, critic, opt_actor, opt_critic, rollout,
    ppo_epochs, mini_batch_size, ppo_clip, value_coef, entropy_coef, device,
):
    """
    PPO update. old_sum_log_probs = log π_T (tempered behavior policy).
    New log-probs at T=1.0 form the ratio π_new / π_T for importance weighting.
    """
    model.train()
    critic.train()

    rewards    = rollout["rewards"]
    values_old = rollout["values"]

    advantages = rewards - values_old
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    returns    = rewards

    N   = rewards.shape[0]
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

            sum_lp_new, entropy, mrna_memory = teacher_force_eval(model, mb_mrna, mb_gen)

            log_ratio   = sum_lp_new - mb_old_slp
            ratio       = torch.exp(log_ratio)
            surr1       = ratio * mb_adv
            surr2       = torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            V_new      = critic(mrna_memory)
            value_loss = F.mse_loss(V_new, mb_ret)

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            opt_actor.zero_grad()
            opt_critic.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.mirna_decoder.parameters(), GRAD_CLIP)
            torch.nn.utils.clip_grad_norm_(model.predictor_head.parameters(), GRAD_CLIP)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), GRAD_CLIP)
            opt_actor.step()
            opt_critic.step()

            # KL in eval mode (removes dropout noise)
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

            if kl > TARGET_KL:
                break

    n = max(1, acc.pop("n"))
    return {k: v / n for k, v in acc.items()}


# =============================================================================
# Validation — Issue 3: critic receives generated tokens; full-buffer rank metric
# =============================================================================

def evaluate_val(model, critic, val_loader, reward_fn, weights, device, n_batches):
    model.eval()
    critic.eval()
    all_rseed, all_rspec_pw, all_rpot, all_noseed = [], [], [], []
    all_off, all_stypes, all_values = [], [], []

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

            r_seed, r_spec_pw, r_pot, seed_types, off_targets, no_seed_mask = reward_fn(
                generated.cpu(), mrna_tokens.cpu()
            )
            values = critic(mrna_memory)

            all_rseed.extend(r_seed.tolist())
            all_rspec_pw.extend(r_spec_pw.tolist())
            all_rpot.extend(r_pot.tolist())
            all_noseed.extend(no_seed_mask.tolist())
            all_off.extend(off_targets)
            all_stypes.extend(seed_types)
            all_values.extend(values.tolist())

    # Rank-based blending for val (consistent with train metric)
    off_tensor   = torch.tensor(all_off, dtype=torch.float)
    N            = len(off_tensor)
    ranks        = off_tensor.argsort().argsort().float()
    r_spec_rank  = 1.0 - 2.0 * ranks / max(1.0, float(N - 1))
    r_spec_blend = ((1.0 - RANK_SPEC_WEIGHT) * torch.tensor(all_rspec_pw)
                    + RANK_SPEC_WEIGHT * r_spec_rank)
    r_total      = combine_rewards(
        torch.tensor(all_rseed), r_spec_blend,
        torch.tensor(all_rpot),  torch.tensor(all_noseed), weights
    )

    cnt   = Counter(all_stypes)
    total = max(1, sum(cnt.values()))
    sdist = {k: cnt.get(k, 0) / total for k in ("8-mer", "7-mer-m8", "7-mer-A1", "6-mer", "none")}

    return {
        "avg_reward":      float(r_total.mean()),
        "avg_off_targets": float(np.mean(all_off)),
        "seed_rate":       1.0 - sdist["none"],
        "8mer_frac":       sdist["8-mer"],
        "7mer_m8_frac":    sdist["7-mer-m8"],
        "7mer_a1_frac":    sdist["7-mer-A1"],
        "6mer_frac":       sdist["6-mer"],
        "avg_value":       float(np.mean(all_values)),
        "n_samples":       total,
    }


# =============================================================================
# Checkpoint helpers
# =============================================================================

def load_pretrained(model, ckpt_path, device):
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
                print(f"[ckpt] Dropping mismatched rotary {k}: {v.shape} vs {sd_model[k].shape}", flush=True)
        else:
            to_load[k] = v
    missing, _ = model.load_state_dict(to_load, strict=False)
    print(f"[ckpt] Loaded {len(to_load)}/{len(sd_ckpt)} tensors. Missing: {len(missing)}", flush=True)
    return model


def get_curriculum(iteration: int):
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
    print("MiRformer-gen PPO RL Finetuning v2 (all 7 issues fixed)", flush=True)
    print(f"  Device:        {DEVICE}", flush=True)
    print(f"  Checkpoint:    {CKPT_PATH}", flush=True)
    print(f"  Save dir:      {RL_CKPT_DIR}", flush=True)
    print(f"  Iterations:    {NUM_ITERATIONS}", flush=True)
    print(f"  Val batches:   {VAL_BATCHES} × {BATCH_SIZE} = {VAL_BATCHES*BATCH_SIZE}", flush=True)
    print(f"  Rank spec wt:  {RANK_SPEC_WEIGHT} (full 512-sample buffer)", flush=True)
    print(f"  Temperature:   {TEMPERATURE_SCHEDULE}", flush=True)
    print(f"  Valid tokens:  EOS + ATCG (EOS bug fixed)", flush=True)
    print(f"  Entropy:       target=0.15 max_coef=0.25 (calibrated)", flush=True)
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

    # Issue 2: freeze mRNA encoder, CNN embedding, AND sn_embedding
    # sn_embedding is shared with frozen encoder — drift would corrupt cross-attn keys
    for p in model.mrna_encoder.parameters():
        p.requires_grad_(False)
    for p in model.cnn_embedding.parameters():
        p.requires_grad_(False)
    for p in model.sn_embedding.parameters():
        p.requires_grad_(False)   # Issue 2: FROZEN (was trainable in RL_finetune.py)
    for p in model.mirna_decoder.parameters():
        p.requires_grad_(True)
    for p in model.predictor_head.parameters():
        p.requires_grad_(True)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable:,} / {n_total:,} (sn_embedding frozen)", flush=True)

    # Issue 3: seed-aware critic
    critic = Critic(hidden_dim=EMBED_DIM).to(device)
    print(f"Critic: mRNA-only MLP (hidden={EMBED_DIM})", flush=True)

    # Optimisers — decoder + predictor_head only (no sn_embedding)
    actor_params = (
        list(model.mirna_decoder.parameters())
        + list(model.predictor_head.parameters())
    )
    opt_actor  = AdamW(actor_params,        lr=LR_ACTOR, weight_decay=0.01)
    opt_critic = AdamW(critic.parameters(), lr=LR_ACTOR, weight_decay=0.01)

    # Reward function
    reward_fn = RewardFunction(KMER_INDEX_PATH)

    # Issue 6: calibrated entropy manager (target=0.15, max_coef=0.25)
    entropy_mgr = EntropyAdaptiveManager(
        initial_coef=0.05,
        target_entropy=0.15,   # Issue 6: was 0.3 (too high for actual H~0.08)
        min_coef=0.01,
        max_coef=0.25,         # Issue 6: was 0.5 (too strong)
        history_len=5,
    )

    # ── Data ─────────────────────────────────────────────────────────────────
    tokenizer    = model.tokenizer
    D_train      = load_dataset(TRAIN_PATH, sep=",")
    D_val        = load_dataset(VALID_PATH, sep=",")
    ds_train     = TargetPredictionDataset(D_train, MRNA_MAX_LEN, MIRNA_MAX_LEN, tokenizer)
    ds_val       = TargetPredictionDataset(D_val,   MRNA_MAX_LEN, MIRNA_MAX_LEN, tokenizer)
    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True,  num_workers=2)
    val_loader   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=2)
    data_cycle   = itertools.cycle(train_loader)
    print(f"Train: {len(ds_train)} samples | Val: {len(ds_val)} samples", flush=True)

    # ── W&B ──────────────────────────────────────────────────────────────────
    wandb.login(key=WANDB_KEY)
    wandb.init(
        project="mirna-RL-finetune",
        name=f"PPO-v2-rank512-mRNACritic-lr{LR_ACTOR:.0e}",
        config={
            "version":          "v2-all-issues-fixed",
            "lr_actor":         LR_ACTOR,
            "ppo_clip":         PPO_CLIP,
            "ppo_epochs":       PPO_EPOCHS,
            "value_coef":       VALUE_COEF,
            "grad_clip":        GRAD_CLIP,
            "target_kl":        TARGET_KL,
            "rollout_batches":  ROLLOUT_BATCHES,
            "batch_size":       BATCH_SIZE,
            "val_batches":      VAL_BATCHES,
            "mini_batch_size":  MINI_BATCH_SIZE,
            "num_iterations":   NUM_ITERATIONS,
            "rank_spec_weight": RANK_SPEC_WEIGHT,
            "rank_scope":       "full_512_buffer",
            "valid_gen_tokens": VALID_GEN_TOKENS,
            "temperature":      TEMPERATURE_SCHEDULE,
            "spec_thresholds":  [SPEC_THRESH_EXCELLENT, SPEC_THRESH_ACCEPTABLE, SPEC_THRESH_POOR],
            "invalid_seed_penalty": INVALID_SEED_PENALTY,
            "sn_embedding":     "frozen",
            "critic":           "mRNA-only MLP (CriticWithSeed reverted)",
            "entropy_target":   0.15,
            "entropy_max_coef": 0.25,
            "issues_fixed":     "1(rank512),2(freeze_emb),3(mRNA-critic-reverted),"
                                "4(IS_logprob),5(val100),6(entropy),7(token_mask+EOS_fix)",
        },
        tags=["RL", "PPO", "v2-fixed", "miRNA-generation", "off-target-reduction"],
        save_code=True,
    )

    # ── Training state ────────────────────────────────────────────────────────
    best_off_targets = float("inf")
    best_state       = None
    patience_counter = 0
    entropy_coef_live = entropy_mgr.coef

    print("Starting PPO RL finetuning v2 (all issues fixed) …", flush=True)

    for iteration in range(NUM_ITERATIONS):
        t0 = time.time()

        weights, entropy_coef_sched = get_curriculum(iteration)
        if iteration in (50, 150):
            entropy_mgr.reset_to(entropy_coef_sched)
            print(f"[iter {iteration+1}] Stage transition: entropy_coef reset to {entropy_coef_sched}", flush=True)

        # ── Collect rollouts ──────────────────────────────────────────────────
        rollout = collect_rollouts(
            model, critic, data_cycle, reward_fn, weights, device,
            n_batches=ROLLOUT_BATCHES,
            iteration=iteration,
        )

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

        if kl > TARGET_KL * 2:
            print(
                f"[iter {iteration+1:3d}] WARNING: high per-token KL={kl:.4f} "
                f"(target={TARGET_KL:.3f}). PPO clip protecting policy.",
                flush=True,
            )

        entropy_coef_live = entropy_mgr.update(entropy, iteration + 1)

        log = {
            "iteration":               iteration + 1,
            "train/avg_reward":        avg_reward,
            "train/avg_off_targets":   avg_off_tgt,
            "train/seed_rate":         seed_rate,
            "train/8mer_frac":         sdist["8-mer"],
            "train/7mer_m8_frac":      sdist["7-mer-m8"],
            "train/7mer_a1_frac":      sdist["7-mer-A1"],
            "train/6mer_frac":         sdist["6-mer"],
            "train/none_frac":         sdist["none"],
            "ppo/policy_loss":         upd["policy_loss"],
            "ppo/value_loss":          upd["value_loss"],
            "ppo/entropy":             entropy,
            "ppo/kl_div":              kl,
            "ppo/clip_frac":           upd["clip_frac"],
            "curriculum/w_seed":       weights["seed"],
            "curriculum/w_spec":       weights["spec"],
            "curriculum/w_pot":        weights["pot"],
            "curriculum/entropy_coef": entropy_coef_live,
            "temperature":             get_temperature(iteration),
            "lr_actor":                opt_actor.param_groups[0]["lr"],
            "time_per_iter_s":         elapsed,
        }

        # ── Validation ────────────────────────────────────────────────────────
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
                "val/n_samples":       vm["n_samples"],
            })
            print(
                f"[iter {iteration+1:3d}] "
                f"R={avg_reward:.4f} | train_off={avg_off_tgt:.0f} "
                f"| seed={seed_rate:.3f} | 8mer={sdist['8-mer']:.3f} "
                f"| KL={kl:.4f} | H={entropy:.3f} "
                f"| val_off={vm['avg_off_targets']:.0f} "
                f"| val_seed={vm['seed_rate']:.3f} | t={elapsed:.0f}s",
                flush=True,
            )

            if vm["avg_off_targets"] < best_off_targets:
                best_off_targets = vm["avg_off_targets"]
                best_state = {
                    "model":       copy.deepcopy(model.state_dict()),
                    "critic":      copy.deepcopy(critic.state_dict()),
                    "iteration":   iteration + 1,
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
            "model":            model.state_dict(),
            "critic":           critic.state_dict(),
            "iteration":        iteration + 1,
            "best_off_targets": best_off_targets,
        },
        final_path,
    )
    print(f"\nTraining complete (v2, all issues fixed).", flush=True)
    print(f"Best val off_targets:  {best_off_targets:.0f}", flush=True)
    print(f"Final model saved to: {final_path}", flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()
