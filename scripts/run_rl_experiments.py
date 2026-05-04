#!/usr/bin/env python3
"""scripts/run_rl_experiments.py

Re-run biological plausibility experiments 1, 2, and 3 using the
RL-finetuned checkpoint (best_off_targets_18137_iter120.pth, v2 run).

Steps
-----
0.  Load RL model; greedy-generate miRNAs for 30nt/100nt/500nt test sets.
    mRNA inputs tokenised with mrna_max_len=120 (truncated/padded, matching RL training).
1.  Exp 1 — Thermodynamic MFE via RNAduplex (same logic as exp1_thermodynamic_affinity.py).
2.  Exp 2 — Transcriptome-wide off-target specificity (same logic as exp2_offtarget_specificity.py).
3.  Exp 3 — Functional potency: Context++ score + cross-attention with RL model.

All outputs saved with "_RLv3" suffix:
  results/exp1_mfe_comparison_RLv3.csv
  results/exp2_off_target_analysis_RLv3.csv
  results/exp3_potency_summary_RLv3.csv
  plots/exp1_mfe_distribution_RLv3.pdf
  plots/exp2_off_target_histogram_RLv3.pdf
  plots/exp2_specificity_comparison_RLv3.pdf
  plots/exp3_seed_type_potency_RLv3.pdf
  plots/exp3_attention_vs_potency_RLv3.pdf
  plots/exp3_lfc_distribution_RLv3.pdf

Usage
-----
    nohup python scripts/run_rl_experiments.py > run_rl_experiments.log 2>&1 &
"""

import os, sys, random, subprocess, pickle, itertools, json
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
from Global_parameters import PROJ_HOME
from utils import load_dataset
from Data_pipeline import TargetPredictionDataset
from DTEA_model import TargetGenerationModel

# ── Paths ──────────────────────────────────────────────────────────────────────
RL_CKPT_PATH = os.path.join(
    PROJ_HOME,
    "checkpoints/TargetScan/TwoTowerTransformer/RL_finetune_v2_mrna_critic",
    "best_off_targets_17778_iter165.pth",
)
KMER_INDEX_PATH = os.path.join(
    PROJ_HOME, "miR_degradome_ago_clip_pairing_data", "hg19_3UTR_kmer_count.pkl"
)
UTR3_FASTA = os.path.join(
    PROJ_HOME, "miR_degradome_ago_clip_pairing_data", "hg19_3UTR_extracted.fa"
)
RNADUPLEX_BIN = "path/to/RNAduplex"

DATA_FILES = {
    "30nt": os.path.join(
        PROJ_HOME, "TargetScan_dataset",
        "generated_mirna_positive_samples_30_randomized_start_test.csv",
    ),
    "100nt": os.path.join(
        PROJ_HOME, "TargetScan_dataset",
        "generated_mirna_positive_primates_test_100_randomized_start_local_self_attn_full_cross_attn.csv",
    ),
    "500nt": os.path.join(
        PROJ_HOME, "TargetScan_dataset",
        "generated_mirna_positive_primates_test_500_randomized_start_local_self_attn_full_cross_attn.csv",
    ),
}

RESULTS_DIR = os.path.join(PROJ_HOME, "results")
PLOTS_DIR   = os.path.join(PROJ_HOME, "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

DEVICE       = "cuda:0"
BATCH_SIZE   = 64
MRNA_MAX_LEN = 120
MIRNA_MAX_LEN = 26    # BOS + 24nt + EOS
MAX_GEN_LEN   = 24

# Token IDs
BOS_ID = 2; EOS_ID = 1; PAD_ID = 4
TOKEN_TO_NT = {7: "A", 8: "T", 9: "C", 10: "G", 11: "N"}

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

BASES_RNA = list("ACGU")
BASES_DNA = list("ACGT")


# =============================================================================
# STEP 0 — Model loading & miRNA generation
# =============================================================================

def load_rl_model(ckpt_path: str, device: str) -> TargetGenerationModel:
    """Load RL-finetuned TargetGenerationModel from checkpoint."""
    model = TargetGenerationModel(
        mirna_max_len  = MIRNA_MAX_LEN,
        mrna_max_len   = MRNA_MAX_LEN,
        embed_dim      = 1024,
        ff_dim         = 4096,
        num_heads      = 8,
        num_layers     = 4,
        batch_size     = BATCH_SIZE,
        vocab_size     = 13,
        n_classes      = 13,
        device         = device,
        use_longformer = True,
        window_size    = 20,
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # RL checkpoint stores model state under the "model" key
    sd_ckpt  = ckpt["model"]
    sd_model = model.state_dict()
    to_load  = {}
    for k, v in sd_ckpt.items():
        if k not in sd_model:
            continue
        if "rotary.cos_emb" in k or "rotary.sin_emb" in k:
            if v.shape == sd_model[k].shape:
                to_load[k] = v
            else:
                print(f"  [ckpt] Dropping mismatched rotary {k}: {v.shape} vs {sd_model[k].shape}")
        else:
            to_load[k] = v
    missing, _ = model.load_state_dict(to_load, strict=False)
    print(f"  [ckpt] Loaded {len(to_load)}/{len(sd_ckpt)} tensors. Missing: {len(missing)}")
    model.to(device)
    model.eval()
    return model


def _encode_mrna(model, mrna_tokens):
    """Frozen mRNA encoder pass → (memory, src_key_mask)."""
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
        memory       = model.mrna_encoder(mrna_emb, mask=lf_mask)
        src_key_mask = src_mask.to(torch.uint8)
    else:
        memory       = model.mrna_encoder(mrna_emb, mask=src_mask)
        src_key_mask = src_mask
    return memory, src_key_mask


def _decode_last_logits(model, tgt_tokens, mrna_memory, src_key_mask):
    """Single decoder step → logits at last position: (B, V)."""
    tgt_mask = model.create_tgt_mask(tgt_tokens).to(tgt_tokens.device)
    tgt_emb  = model.sn_embedding(tgt_tokens)
    out      = model.mirna_decoder(
        x=tgt_emb, memory=mrna_memory, src_mask=src_key_mask, tgt_mask=tgt_mask
    )
    return model.predictor_head(out[:, -1, :])


def _tokens_to_dna_3to5(token_ids) -> str:
    """Tokens → 3'→5' DNA string (skips BOS; stops at EOS/PAD)."""
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


def generate_rl_mirnas(model, csv_path: str, device: str) -> list[str]:
    """
    Load test CSV, tokenise mRNAs (mrna_max_len=120), greedy-decode with RL model.
    Returns list of generated miRNA strings in 3'→5' DNA order (matching CSV convention).
    Order is preserved (no shuffle).
    """
    tokenizer = model.tokenizer
    D = load_dataset(csv_path, sep=",")
    ds = TargetPredictionDataset(D, MRNA_MAX_LEN, MIRNA_MAX_LEN, tokenizer)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=2)

    generated_mirnas = []
    n_total = len(ds)
    n_done  = 0
    with torch.no_grad():
        for batch in loader:
            mrna_tokens = batch["mrna_input_ids"].to(device)
            B = mrna_tokens.size(0)

            memory, src_key_mask = _encode_mrna(model, mrna_tokens)
            generated = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
            finished  = torch.zeros(B, dtype=torch.bool, device=device)

            for _ in range(MAX_GEN_LEN):
                logits  = _decode_last_logits(model, generated, memory, src_key_mask)
                next_id = logits.argmax(dim=-1)
                next_id = torch.where(finished, torch.full_like(next_id, PAD_ID), next_id)
                generated = torch.cat([generated, next_id.unsqueeze(1)], dim=1)
                finished  = finished | (next_id == EOS_ID)
                if finished.all():
                    break

            for seq in generated.cpu():
                generated_mirnas.append(_tokens_to_dna_3to5(seq))
            n_done += B
            if n_done % 512 == 0 or n_done >= n_total:
                print(f"    Generated {min(n_done, n_total)}/{n_total}", flush=True)

    return generated_mirnas


# =============================================================================
# STEP 1 — Experiment 1: Thermodynamic MFE
# =============================================================================

def _to_rna(seq: str) -> str:
    return seq.upper().replace("T", "U")

def _rc_rna(seq: str) -> str:
    comp = {"A": "U", "U": "A", "C": "G", "G": "C", "N": "N", "T": "A"}
    return "".join(comp.get(b, "N") for b in seq.upper()[::-1])

def _dinucleotide_shuffle(seq: str, rng=None) -> str:
    if rng is None: rng = random
    seq = seq.upper(); n = len(seq)
    if n <= 2:
        s = list(seq); rng.shuffle(s); return "".join(s)
    adj: dict = defaultdict(list)
    for i in range(n - 1):
        adj[seq[i]].append(seq[i + 1])
    for k in adj:
        rng.shuffle(adj[k])
    start = seq[0]; path = []; stack = [start]
    local = {k: list(v) for k, v in adj.items()}
    while stack:
        v = stack[-1]
        if local.get(v):
            stack.append(local[v].pop())
        else:
            path.append(stack.pop())
    result = path[::-1]
    if len(result) != n:
        s = list(seq); rng.shuffle(s); return "".join(s)
    return "".join(result)

def _make_rand_seed_mirna_rna(mirna_len: int, mrna_seq: str, seed_s: int, seed_e: int) -> str:
    mrna_seed   = _to_rna(mrna_seq[seed_s:seed_e])
    mirna_seed  = _rc_rna(mrna_seed)
    pre   = [random.choice(BASES_RNA) for _ in range(1)]
    post  = [random.choice(BASES_RNA) for _ in range(max(mirna_len - 1 - len(mirna_seed), 0))]
    result = (pre + list(mirna_seed) + post)[:mirna_len]
    if len(result) < mirna_len:
        result += [random.choice(BASES_RNA)] * (mirna_len - len(result))
    return "".join(result)

def _parse_rnaduplex(line: str) -> float:
    line = line.strip()
    if not line: return np.nan
    try:
        return float(line.split()[-1].strip("()"))
    except (ValueError, IndexError):
        return np.nan

def _compute_mfe_batch(seq_pairs: list) -> list:
    if not seq_pairs: return []
    stdin_lines = []
    for s1, s2 in seq_pairs:
        stdin_lines.append(_to_rna(s1))
        stdin_lines.append(_to_rna(s2))
    stdin_text = "\n".join(stdin_lines) + "\n@\n"
    try:
        result = subprocess.run(
            [RNADUPLEX_BIN, "--noLP"],
            input=stdin_text, capture_output=True, text=True, timeout=600,
        )
        out = [l for l in result.stdout.splitlines() if l.strip()]
        mfes = [_parse_rnaduplex(l) for l in out]
        while len(mfes) < len(seq_pairs):
            mfes.append(np.nan)
        return mfes[:len(seq_pairs)]
    except Exception as e:
        print(f"  [WARNING] RNAduplex failed: {e}")
        return [np.nan] * len(seq_pairs)

def _wilcoxon_less(a, b):
    a_c = a.dropna().values; b_c = b.dropna().values
    if len(a_c) < 5 or len(b_c) < 5: return np.nan, np.nan
    return stats.mannwhitneyu(a_c, b_c, alternative="less")

def run_exp1(all_dfs_rl: dict) -> pd.DataFrame:
    """Run MFE experiment on RL-generated miRNAs."""
    BATCH = 200
    all_rows = []

    for name, df in all_dfs_rl.items():
        print(f"\n[Exp1] {name}: {len(df)} pairs", flush=True)
        gen_pairs = []; real_pairs = []; rand_pairs = []; shuf_pairs = []
        rand_mirnas = []

        for _, row in df.iterrows():
            mrna       = str(row["mRNA sequence"])
            gen_mirna  = str(row["rl_generated_mirna"])
            real_mirna = str(row["miRNA sequence"])
            seed_s     = int(row["seed start"]); seed_e = int(row["seed end"])

            gen_pairs.append((gen_mirna, mrna))
            real_pairs.append((real_mirna, mrna))

            rand = _make_rand_seed_mirna_rna(len(gen_mirna), mrna, seed_s, seed_e)
            rand_pairs.append((rand, mrna))
            rand_mirnas.append(rand)
            shuf_pairs.append((gen_mirna, _dinucleotide_shuffle(mrna)))

        def _batch_mfe(pairs, label):
            mfes = []
            for start in range(0, len(pairs), BATCH):
                if start % 1000 == 0:
                    print(f"  [{label}] {start}/{len(pairs)} ...", flush=True)
                mfes.extend(_compute_mfe_batch(pairs[start:start + BATCH]))
            return mfes

        mfe_gen  = _batch_mfe(gen_pairs,  "generated-RL")
        mfe_real = _batch_mfe(real_pairs, "real")
        mfe_rand = _batch_mfe(rand_pairs, "random")
        mfe_shuf = _batch_mfe(shuf_pairs, "shuffled")

        sub = df[["Transcript ID", "miRNA ID", "seed start", "seed end"]].copy()
        sub["mRNA_length"]           = name
        sub["generated_mirna"]       = [p[0] for p in gen_pairs]
        sub["real_mirna"]            = [p[0] for p in real_pairs]
        sub["random_seed_mirna"]     = rand_mirnas
        sub["MFE_generated"]         = mfe_gen
        sub["MFE_real_mirna"]        = mfe_real
        sub["MFE_random_seed_match"] = mfe_rand
        sub["MFE_shuffled_mRNA"]     = mfe_shuf
        all_rows.append(sub)

    return pd.concat(all_rows, ignore_index=True)


def plot_exp1_mfe(all_df: pd.DataFrame, out_pdf: str) -> None:
    lengths = ["30nt", "100nt", "500nt"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colours = {
        "MFE_generated":         ("royalblue",   "Generated miRNA (RL)"),
        "MFE_real_mirna":        ("forestgreen", "Real miRNA (GT)"),
        "MFE_random_seed_match": ("darkorange",  "Random (seed-match ctrl)"),
        "MFE_shuffled_mRNA":     ("firebrick",   "Shuffled mRNA ctrl"),
    }
    for ax, name in zip(axes, lengths):
        sub = all_df[all_df["mRNA_length"] == name]
        for col, (colour, label) in colours.items():
            vals = sub[col].dropna()
            ax.hist(vals, bins=60, alpha=0.55, color=colour, label=label, density=True)
            ax.axvline(vals.median(), color=colour, linestyle="--", linewidth=1.2, alpha=0.85)
        stat, pval = _wilcoxon_less(sub["MFE_generated"], sub["MFE_random_seed_match"])
        pval_str = f"p={pval:.2e}" if not np.isnan(pval) else "p=N/A"
        ax.set_xlabel("MFE (kcal/mol)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"{name} mRNA\nGenerated(RL) vs Random ctrl: {pval_str}", fontsize=12)
        ax.legend(fontsize=8)
    plt.suptitle("Experiment 1 (RL-finetuned) – Thermodynamic Binding Affinity (ΔG)",
                 fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {out_pdf}", flush=True)


def print_exp1_stats(all_df: pd.DataFrame) -> dict:
    print("\n" + "="*65)
    print("EXP 1 — MFE STATISTICAL SUMMARY (RL model)")
    print("="*65)
    stats_out = {}
    for name in ["30nt", "100nt", "500nt"]:
        sub = all_df[all_df["mRNA_length"] == name]
        print(f"\n{name} (n={len(sub)}):")
        row_stats = {}
        for col in ["MFE_generated", "MFE_real_mirna", "MFE_random_seed_match", "MFE_shuffled_mRNA"]:
            v = sub[col].dropna()
            print(f"  {col:35s}: mean={v.mean():.3f}  median={v.median():.3f}  std={v.std():.3f}")
            row_stats[col] = {"mean": float(v.mean()), "median": float(v.median()), "std": float(v.std())}
        stat, pval = _wilcoxon_less(sub["MFE_generated"], sub["MFE_random_seed_match"])
        stat2, pval2 = _wilcoxon_less(sub["MFE_generated"], sub["MFE_shuffled_mRNA"])
        print(f"  Wilcoxon (generated < random seed-match): U={stat:.0f}, p={pval:.4e}")
        print(f"  Wilcoxon (generated < shuffled mRNA):     U={stat2:.0f}, p={pval2:.4e}")
        row_stats["wilcoxon_vs_rand"] = {"U": float(stat), "p": float(pval)}
        row_stats["wilcoxon_vs_shuf"] = {"U": float(stat2), "p": float(pval2)}
        stats_out[name] = row_stats
    return stats_out


# =============================================================================
# STEP 2 — Experiment 2: Off-Target Specificity
# =============================================================================

def _dna_complement(seq: str) -> str:
    c = {"A": "T", "C": "G", "G": "C", "T": "A", "U": "A", "N": "N"}
    return "".join(c.get(b, "N") for b in seq.upper())

def _to_dna(seq: str) -> str:
    return seq.upper().replace("U", "T")

def _extract_seed_patterns(mirna: str) -> dict:
    """mirna in 3'→5' storage order → return DNA strings to search in mRNA."""
    fwd = _to_dna(mirna[::-1])
    seed7 = fwd[1:8]; seed7_rc = _dna_complement(seed7)[::-1]
    seed6 = fwd[1:7]; seed6_rc = _dna_complement(seed6)[::-1]
    return {
        "8-mer":    seed7_rc + "A",
        "7-mer-m8": seed7_rc,
        "7-mer-A1": seed6_rc + "A",
        "6-mer":    seed6_rc,
    }

def _load_fasta(fasta_path: str) -> dict:
    import gzip
    seqs = {}
    opener = gzip.open if fasta_path.endswith(".gz") else open
    cur_id, buf = None, []
    with opener(fasta_path, "rt") as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if cur_id: seqs[cur_id] = "".join(buf).upper()
                cur_id = line[1:].split()[0].split(".")[0]; buf = []
            else:
                buf.append(line)
    if cur_id: seqs[cur_id] = "".join(buf).upper()
    print(f"  Loaded {len(seqs)} sequences from {fasta_path}", flush=True)
    return seqs

def _count_off_targets(mirna: str, true_tx_id: str, count_index: dict, utr3_seqs: dict):
    patterns  = _extract_seed_patterns(mirna)
    true_base = true_tx_id.split(".")[0]
    true_utr  = utr3_seqs.get(true_base, "")
    def adj(k, mtype):
        pat = patterns[mtype]; cnt = count_index[k].get(pat, 0)
        if pat in true_utr: cnt = max(0, cnt - 1)
        return cnt
    n_8   = adj(8, "8-mer")
    n_7m8 = adj(7, "7-mer-m8")
    n_7a1 = adj(7, "7-mer-A1")
    total = adj(6, "6-mer")
    return total, n_8, n_7m8, n_7a1

def _make_rand_seed_mirna_dna(mirna_len: int, mrna: str, seed_s: int, seed_e: int) -> str:
    mrna_seed  = _to_dna(mrna)[seed_s:seed_e]
    mirna_seed = _dna_complement(mrna_seed)[::-1]
    pre  = [random.choice(BASES_DNA) for _ in range(1)]
    post = [random.choice(BASES_DNA) for _ in range(max(mirna_len - 1 - len(mirna_seed), 0))]
    result = (pre + list(mirna_seed) + post)[:mirna_len]
    if len(result) < mirna_len:
        result += [random.choice(BASES_DNA)] * (mirna_len - len(result))
    return "".join(result)

def run_exp2(all_dfs_rl: dict, count_index: dict, utr3_seqs: dict) -> pd.DataFrame:
    all_rows = []
    for name, df in all_dfs_rl.items():
        print(f"\n[Exp2] {name}: {len(df)} pairs", flush=True)
        for i, row in df.iterrows():
            if i % 1000 == 0:
                print(f"  {i}/{len(df)} ...", flush=True)
            mrna       = str(row["mRNA sequence"])
            gen_mirna  = str(row["rl_generated_mirna"])
            real_mirna = str(row["miRNA sequence"])
            tx_id      = str(row["Transcript ID"])
            seed_s     = int(row["seed start"]); seed_e = int(row["seed end"])
            rand_mirna = _make_rand_seed_mirna_dna(len(gen_mirna), mrna, seed_s, seed_e)

            gen_ot,  g8, g7m, g7a = _count_off_targets(gen_mirna,  tx_id, count_index, utr3_seqs)
            real_ot, r8, r7m, r7a = _count_off_targets(real_mirna, tx_id, count_index, utr3_seqs)
            rand_ot, _,  _,   _   = _count_off_targets(rand_mirna, tx_id, count_index, utr3_seqs)

            all_rows.append({
                "mRNA_length":           name,
                "Transcript_ID":         tx_id,
                "miRNA_ID":              row["miRNA ID"],
                "gen_off_target_count":  gen_ot,
                "gen_8mer_off":          g8,
                "gen_7m8_off":           g7m,
                "gen_7a1_off":           g7a,
                "real_off_target_count": real_ot,
                "real_8mer_off":         r8,
                "real_7m8_off":          r7m,
                "real_7a1_off":          r7a,
                "rand_off_target_count": rand_ot,
                "gen_si":   1 / (gen_ot  + 1),
                "real_si":  1 / (real_ot + 1),
                "rand_si":  1 / (rand_ot + 1),
            })
    return pd.DataFrame(all_rows)

def plot_exp2_histogram(all_df: pd.DataFrame, out_pdf: str) -> None:
    lengths = ["30nt", "100nt", "500nt"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, name in zip(axes, lengths):
        sub = all_df[all_df["mRNA_length"] == name]
        mx = max(sub["gen_off_target_count"].max(), sub["real_off_target_count"].max(),
                 sub["rand_off_target_count"].max())
        bins = min(80, int(mx) + 2)
        ax.hist(sub["gen_off_target_count"],  bins=bins, alpha=0.6,
                color="royalblue",  label="Generated miRNA (RL)", density=True)
        ax.hist(sub["real_off_target_count"], bins=bins, alpha=0.6,
                color="forestgreen", label="Real miRNA (GT)", density=True)
        ax.hist(sub["rand_off_target_count"], bins=bins, alpha=0.6,
                color="darkorange",  label="Random seed-match", density=True)
        u, p = stats.mannwhitneyu(sub["gen_off_target_count"], sub["real_off_target_count"],
                                   alternative="two-sided")
        ax.set_title(f"{name}\nGen(RL) vs Real p={p:.2e}", fontsize=11)
        ax.set_xlabel("Off-target transcript count", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(fontsize=8)
    plt.suptitle("Experiment 2 (RL-finetuned) – Transcriptome-wide Off-Target Count",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {out_pdf}", flush=True)

def plot_exp2_specificity(all_df: pd.DataFrame, out_pdf: str) -> None:
    lengths = ["30nt", "100nt", "500nt"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    colours = ["royalblue", "forestgreen", "darkorange"]
    for ax, name in zip(axes, lengths):
        sub  = all_df[all_df["mRNA_length"] == name]
        data = [sub["gen_si"].values, sub["real_si"].values, sub["rand_si"].values]
        bp   = ax.boxplot(data, patch_artist=True)
        for patch, colour in zip(bp["boxes"], colours):
            patch.set_facecolor(colour)
        ax.set_xticklabels(["Generated (RL)", "Real miRNA", "Random"], fontsize=9)
        ax.set_title(f"{name}", fontsize=12)
        if ax == axes[0]:
            ax.set_ylabel("Specificity Index (1/(off-targets+1))", fontsize=11)
    plt.suptitle("Experiment 2 (RL-finetuned) – Specificity Index Comparison",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {out_pdf}", flush=True)

def print_exp2_stats(all_df: pd.DataFrame) -> dict:
    print("\n" + "="*65)
    print("EXP 2 — OFF-TARGET STATISTICAL SUMMARY (RL model)")
    print("="*65)
    stats_out = {}
    for name in ["30nt", "100nt", "500nt"]:
        sub = all_df[all_df["mRNA_length"] == name]
        print(f"\n{name} (n={len(sub)}):")
        row_stats = {}
        for col in ["gen_off_target_count", "real_off_target_count", "rand_off_target_count"]:
            v = sub[col]
            print(f"  {col:30s}: mean={v.mean():.1f}  median={v.median():.0f}  "
                  f"mean_SI={1/(v.mean()+1):.4f}")
            row_stats[col] = {"mean": float(v.mean()), "median": float(v.median())}
        u, pv = stats.mannwhitneyu(sub["gen_off_target_count"], sub["real_off_target_count"],
                                   alternative="two-sided")
        u2, pv2 = stats.mannwhitneyu(sub["gen_off_target_count"], sub["rand_off_target_count"],
                                     alternative="two-sided")
        print(f"  Wilcoxon (gen vs real):   U={u:.0f}, p={pv:.4e}")
        print(f"  Wilcoxon (gen vs random): U={u2:.0f}, p={pv2:.4e}")
        row_stats["wilcoxon_vs_real"] = {"U": float(u), "p": float(pv)}
        row_stats["wilcoxon_vs_rand"] = {"U": float(u2), "p": float(pv2)}
        stats_out[name] = row_stats
    return stats_out


# =============================================================================
# STEP 3 — Experiment 3: Functional Potency
# =============================================================================

SITE_TYPE_SCORE = {
    "8-mer": -0.310, "7-mer-m8": -0.161, "7-mer-A1": -0.099, "6-mer": -0.015, None: 0.0,
}
W_LOCAL_AU = -0.380
W_3PRIME   = -0.095

def _canonical_seed_match(mrna: str, mirna: str):
    mrna  = _to_dna(mrna); mirna = _to_dna(mirna)
    mirna_fwd = mirna[::-1]
    seed = mirna_fwd[1:8]; seed_rc = _dna_complement(seed)[::-1]
    if (t := seed_rc + "A") in mrna:    return "8-mer",    mrna.find(t)
    if seed_rc in mrna:                  return "7-mer-m8", mrna.find(seed_rc)
    seed6 = mirna_fwd[1:7]; s6rc = _dna_complement(seed6)[::-1]
    if (t2 := s6rc + "A") in mrna:      return "7-mer-A1", mrna.find(t2)
    if s6rc in mrna:                     return "6-mer",    mrna.find(s6rc)
    return None, -1

def _local_au(mrna: str, seed_pos: int, flank: int = 30) -> float:
    lo = max(0, seed_pos - flank); hi = min(len(mrna), seed_pos + flank)
    win = mrna[lo:hi].upper().replace("T", "U")
    if not win: return 0.5
    return sum(1 for c in win if c in "AU") / len(win)

def _three_prime_pairs(mirna: str, mrna: str, seed_pos: int) -> int:
    mirna_fwd = _to_dna(mirna[::-1]).upper(); mrna_dna = _to_dna(mrna).upper()
    supp = mirna_fwd[12:16]
    if len(supp) < 4 or seed_pos < 4: return 0
    mrna_tgt = mrna_dna[max(0, seed_pos - 4):seed_pos][::-1]
    if len(mrna_tgt) < len(supp): return 0
    wc = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return sum(1 for m, r in zip(supp, mrna_tgt) if wc.get(m) == r)

def _ctx_score(seed_type, au: float, pairs: int) -> float:
    return SITE_TYPE_SCORE[seed_type] + W_LOCAL_AU * au + W_3PRIME * pairs

def run_exp3_part_a(all_dfs_rl: dict) -> pd.DataFrame:
    all_rows = []
    for name, df in all_dfs_rl.items():
        print(f"\n[Exp3A] {name}: {len(df)} pairs", flush=True)
        for _, row in df.iterrows():
            mrna       = str(row["mRNA sequence"])
            gen_mirna  = str(row["rl_generated_mirna"])
            real_mirna = str(row["miRNA sequence"])

            gen_st,  gen_sp  = _canonical_seed_match(mrna, gen_mirna)
            gen_au   = _local_au(mrna, gen_sp)   if gen_sp  >= 0 else 0.5
            gen_pairs= _three_prime_pairs(gen_mirna,  mrna, gen_sp)  if gen_sp  >= 0 else 0
            gen_score= _ctx_score(gen_st,  gen_au,  gen_pairs)

            real_st, real_sp = _canonical_seed_match(mrna, real_mirna)
            real_au  = _local_au(mrna, real_sp)  if real_sp >= 0 else 0.5
            real_p   = _three_prime_pairs(real_mirna, mrna, real_sp) if real_sp >= 0 else 0
            real_score = _ctx_score(real_st, real_au, real_p)

            all_rows.append({
                "mRNA_length":         name,
                "Transcript_ID":       row["Transcript ID"],
                "miRNA_ID":            row["miRNA ID"],
                "gen_seed_type":       gen_st if gen_st else "None",
                "gen_seed_pos":        gen_sp,
                "gen_local_au":        gen_au,
                "gen_3prime_pairs":    gen_pairs,
                "gen_context_score":   gen_score,
                "real_seed_type":      real_st if real_st else "None",
                "real_seed_pos":       real_sp,
                "real_local_au":       real_au,
                "real_3prime_pairs":   real_p,
                "real_context_score":  real_score,
            })
    return pd.DataFrame(all_rows)


def run_exp3_part_b(model, df_100nt: pd.DataFrame, device: str) -> np.ndarray:
    """Cross-attention extraction for 100nt test set using RL model."""
    mrna_seqs = df_100nt["mRNA sequence"].tolist()
    tokenizer = model.tokenizer
    PAD = tokenizer.pad_token_id; EOS = tokenizer.eos_token_id; BOS = tokenizer.bos_token_id

    n = len(mrna_seqs); all_seed_attn = []
    SEED_STEPS = list(range(1, 8))

    for start in range(0, n, BATCH_SIZE):
        end   = min(start + BATCH_SIZE, n)
        batch = mrna_seqs[start:end]; bsz = len(batch)

        mrna_ids_list = []
        for seq in batch:
            enc = tokenizer(seq.upper(), add_special_tokens=False,
                            max_length=MRNA_MAX_LEN - 1, truncation=True, padding=False)
            ids = enc["input_ids"] + [EOS]
            ids = (ids + [PAD] * MRNA_MAX_LEN)[:MRNA_MAX_LEN]
            mrna_ids_list.append(ids)

        mrna_tokens = torch.tensor(mrna_ids_list, dtype=torch.long, device=device)
        mrna_mask   = (mrna_tokens != PAD).long()

        with torch.no_grad():
            mrna_emb = model.sn_embedding(mrna_tokens)
            mrna_cnn = model.cnn_embedding(mrna_emb.transpose(-1, -2))
            mrna_emb = mrna_emb + mrna_cnn
            lf_mask  = torch.where(mrna_mask > 0, torch.zeros_like(mrna_mask),
                                   torch.full_like(mrna_mask, -1))
            memory   = model.mrna_encoder(mrna_emb, mask=lf_mask)

        sample_attn = np.zeros((bsz, MRNA_MAX_LEN), dtype=np.float32)
        step_counts = np.zeros(bsz, dtype=np.int32)
        tgt_ids     = torch.full((bsz, 1), BOS, dtype=torch.long, device=device)

        for step in range(MIRNA_MAX_LEN - 1):
            with torch.no_grad():
                t        = tgt_ids.size(1)
                tgt_emb  = model.sn_embedding(tgt_ids)
                tgt_mask = torch.tril(torch.ones(t, t, device=device)).unsqueeze(0)
                x        = tgt_emb
                for layer in model.mirna_decoder.layers:
                    sa_out = layer.self_attn(x, x, x, mask=tgt_mask)
                    x      = layer.norm1(x + layer.dropout(sa_out))
                    ca_out = layer.cross_attn(x, memory, memory, mask=mrna_mask)
                    x      = layer.norm2(x + layer.dropout(ca_out))
                    x      = layer.norm3(x + layer.dropout(layer.feed_forward(x)))

                last_ca = model.mirna_decoder.layers[-1].cross_attn.last_attention
                attn_mean = last_ca[:, :, -1, :].cpu().numpy().mean(axis=1)  # (B, mRNA_len)
                if step in SEED_STEPS:
                    sample_attn += attn_mean; step_counts += 1

                out      = model.mirna_decoder.norm(x)
                out      = model.mirna_decoder.dropout(out)
                logits   = model.predictor_head(out[:, -1, :])
                next_tok = logits.argmax(dim=-1, keepdim=True)
                tgt_ids  = torch.cat([tgt_ids, next_tok], dim=1)
                if (next_tok.squeeze(-1) == EOS).all():
                    break

        for i in range(bsz):
            if step_counts[i] > 0:
                sample_attn[i] /= step_counts[i]
        all_seed_attn.append(sample_attn)
        if start % 512 == 0:
            print(f"  Cross-attn: {end}/{n}", flush=True)

    return np.concatenate(all_seed_attn, axis=0)  # (N, mRNA_len)


def plot_exp3_seed_potency(all_df: pd.DataFrame, out_pdf: str) -> None:
    lengths = ["30nt", "100nt", "500nt"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    stypes  = ["8-mer", "7-mer-m8", "7-mer-A1", "6-mer", "None"]
    colours = {"8-mer": "#1a6faf", "7-mer-m8": "#57a0d2",
               "7-mer-A1": "#f4a460", "6-mer": "#e07b39", "None": "#bbb"}
    for ax, name in zip(axes, lengths):
        sub  = all_df[all_df["mRNA_length"] == name]
        data = [sub.loc[sub["gen_seed_type"] == st, "gen_context_score"].dropna().values
                for st in stypes]
        labels = [f"{st}\n(n={len(d)})" for st, d in zip(stypes, data)]
        valid  = [(d, l) for d, l in zip(data, labels) if len(d) > 0]
        if valid:
            bp = ax.boxplot([d for d, _ in valid], patch_artist=True, vert=True)
            ax.set_xticklabels([l for _, l in valid], rotation=20, ha="right", fontsize=9)
            for patch, (_, l) in zip(bp["boxes"], valid):
                st_name = l.split("\n")[0]
                patch.set_facecolor(colours.get(st_name, "#ccc"))
        ax.set_title(f"{name} mRNA (RL)", fontsize=12)
        ax.set_xlabel("Seed Type", fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel("Simplified Context++ Score", fontsize=11)
    plt.suptitle("Experiment 3 (RL-finetuned) – Predicted Potency by Seed Type",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight"); plt.close()
    print(f"Plot saved → {out_pdf}", flush=True)

def plot_exp3_attn_potency(attn: np.ndarray, scores: np.ndarray, out_pdf: str) -> None:
    """Scatter + marginal histograms: cross-attention vs Context++ score.

    Cross-attention distribution (x-axis marginal) → orange (#e07b39).
    Context++ score distribution (y-axis marginal) → blue (#1a6faf).
    """
    CLR_ATTN  = "#e07b39"   # orange  — cross-attention
    CLR_SCORE = "#1a6faf"   # blue    — Context++ score

    mask = np.isfinite(attn) & np.isfinite(scores)
    x = attn[mask]; y = scores[mask]
    r, pval = stats.pearsonr(x, y)
    rho, pval_sp = stats.spearmanr(x, y)

    # Joint-distribution layout: main scatter + top/right marginal histograms
    fig = plt.figure(figsize=(8, 7))
    gs  = fig.add_gridspec(
        2, 2,
        width_ratios=(5, 1.2), height_ratios=(1.2, 5),
        hspace=0.05, wspace=0.05,
    )
    ax_main  = fig.add_subplot(gs[1, 0])
    ax_top   = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Main scatter
    ax_main.scatter(x, y, alpha=0.20, s=8, color="#888888", edgecolors="none")
    m, b = np.polyfit(x, y, 1)
    xr = np.linspace(x.min(), x.max(), 200)
    ax_main.plot(xr, m * xr + b, color="black", linewidth=1.8)
    ax_main.set_xlabel("Mean Cross-Attention at Seed Steps (RL model)", fontsize=11)
    ax_main.set_ylabel("Simplified Context++ Score",                     fontsize=11)

    # Top marginal: cross-attention distribution (orange)
    ax_top.hist(x, bins=50, color=CLR_ATTN, alpha=0.85, density=True)
    ax_top.axvline(np.mean(x), color=CLR_ATTN, linestyle="--", linewidth=1.5,
                   label=f"mean={np.mean(x):.4f}")
    ax_top.set_ylabel("Density", fontsize=9)
    ax_top.legend(fontsize=8, loc="upper right")
    plt.setp(ax_top.get_xticklabels(), visible=False)

    # Right marginal: Context++ score distribution (blue)
    ax_right.hist(y, bins=50, color=CLR_SCORE, alpha=0.85, density=True,
                  orientation="horizontal")
    ax_right.axhline(np.mean(y), color=CLR_SCORE, linestyle="--", linewidth=1.5,
                     label=f"mean={np.mean(y):.3f}")
    ax_right.set_xlabel("Density", fontsize=9)
    ax_right.legend(fontsize=8, loc="upper right")
    plt.setp(ax_right.get_yticklabels(), visible=False)

    fig.suptitle(
        f"Cross-Attention vs Potency (100nt, RL)\n"
        f"Pearson r={r:.3f} (p={pval:.2e}),  Spearman ρ={rho:.3f} (p={pval_sp:.2e})",
        fontsize=11, y=1.01,
    )
    plt.savefig(out_pdf, bbox_inches="tight"); plt.close()
    print(f"Plot saved → {out_pdf}", flush=True)
    return r, pval, rho, pval_sp

def plot_exp3_lfc(all_df: pd.DataFrame, out_pdf: str) -> None:
    lengths = ["30nt", "100nt", "500nt"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, name in zip(axes, lengths):
        sub = all_df[all_df["mRNA_length"] == name]
        ax.hist(sub["gen_context_score"].dropna(),  bins=40, alpha=0.6,
                color="royalblue",  label="Generated miRNA (RL)", density=True)
        ax.hist(sub["real_context_score"].dropna(), bins=40, alpha=0.6,
                color="forestgreen", label="Real miRNA (GT)", density=True)
        ax.axvline(sub["gen_context_score"].median(),  color="royalblue",  linestyle="--", lw=1.5)
        ax.axvline(sub["real_context_score"].median(), color="forestgreen", linestyle="--", lw=1.5)
        u, pval = stats.mannwhitneyu(sub["gen_context_score"].dropna(),
                                      sub["real_context_score"].dropna(),
                                      alternative="two-sided")
        ax.set_title(f"{name}\nGen(RL) vs Real p={pval:.2e}", fontsize=11)
        ax.set_xlabel("Simplified Context++ Score", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(fontsize=8)
    plt.suptitle("Experiment 3 (RL-finetuned) – Predicted Log Fold Change Distribution",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight"); plt.close()
    print(f"Plot saved → {out_pdf}", flush=True)

def print_exp3_stats(all_df: pd.DataFrame) -> dict:
    print("\n" + "="*65)
    print("EXP 3 — POTENCY STATISTICAL SUMMARY (RL model)")
    print("="*65)
    stats_out = {}
    for name in ["30nt", "100nt", "500nt"]:
        sub = all_df[all_df["mRNA_length"] == name]
        print(f"\n{name} (n={len(sub)}):")
        cnt = Counter(sub["gen_seed_type"])
        tot = max(1, sum(cnt.values()))
        print(f"  Seed types: { {k: f'{v/tot*100:.1f}%' for k, v in cnt.most_common()} }")
        gen_score  = sub["gen_context_score"].dropna()
        real_score = sub["real_context_score"].dropna()
        print(f"  gen  context++: mean={gen_score.mean():.4f}  median={gen_score.median():.4f}")
        print(f"  real context++: mean={real_score.mean():.4f}  median={real_score.median():.4f}")
        seed_rate = 1 - (cnt.get("None", 0) / tot)
        print(f"  Seed match rate: {seed_rate:.3f}")
        stats_out[name] = {
            "seed_types": {k: float(v/tot) for k, v in cnt.items()},
            "seed_match_rate": float(seed_rate),
            "gen_context_score_mean":   float(gen_score.mean()),
            "gen_context_score_median": float(gen_score.median()),
            "real_context_score_mean":  float(real_score.mean()),
        }
    return stats_out


# =============================================================================
# Main
# =============================================================================

def main():
    device = DEVICE
    print("=" * 70, flush=True)
    print("MiRformer RL Experiments (v3 checkpoint: iter 165, val_off=17778)", flush=True)
    print(f"  Checkpoint: {RL_CKPT_PATH}", flush=True)
    print("=" * 70, flush=True)

    # ── STEP 0: Load RL model ──────────────────────────────────────────────────
    print("\n[Step 0] Loading RL model ...", flush=True)
    model = load_rl_model(RL_CKPT_PATH, device)

    # ── STEP 0b: Generate miRNAs for each test set ─────────────────────────────
    all_dfs_rl = {}
    for name, csv_path in DATA_FILES.items():
        print(f"\n  Generating RL miRNAs for {name} ...", flush=True)
        df = pd.read_csv(csv_path)
        rl_mirnas = generate_rl_mirnas(model, csv_path, device)
        assert len(rl_mirnas) == len(df), (
            f"Length mismatch: {len(rl_mirnas)} vs {len(df)}"
        )
        df["rl_generated_mirna"] = rl_mirnas
        all_dfs_rl[name] = df
        print(f"  {name}: generated {len(rl_mirnas)} miRNAs", flush=True)
        # Example
        for i in range(3):
            orig = df.iloc[i]["generated_mirna"]
            rl   = rl_mirnas[i]
            print(f"    Sample {i}: orig={orig}  RL={rl}", flush=True)

    # ── STEP 1: Experiment 1 — MFE ────────────────────────────────────────────
    print("\n" + "=" * 70, flush=True)
    print("[Step 1] Experiment 1 — Thermodynamic MFE", flush=True)
    print("=" * 70, flush=True)

    exp1_df = run_exp1(all_dfs_rl)
    csv1 = os.path.join(RESULTS_DIR, "exp1_mfe_comparison_RLv3.csv")
    exp1_df.to_csv(csv1, index=False)
    print(f"\nExp1 CSV → {csv1}", flush=True)

    exp1_stats = print_exp1_stats(exp1_df)
    plot_exp1_mfe(exp1_df, os.path.join(PLOTS_DIR, "exp1_mfe_distribution_RLv3.pdf"))

    # ── STEP 2: Experiment 2 — Off-Target ─────────────────────────────────────
    print("\n" + "=" * 70, flush=True)
    print("[Step 2] Experiment 2 — Off-Target Specificity", flush=True)
    print("=" * 70, flush=True)

    print("  Loading 3'UTR FASTA cache ...", flush=True)
    utr3_seqs = _load_fasta(UTR3_FASTA)
    print("  Loading k-mer count index ...", flush=True)
    with open(KMER_INDEX_PATH, "rb") as fh:
        count_index = pickle.load(fh)
    print(f"  k-mer index loaded (keys: {list(count_index.keys())})", flush=True)

    exp2_df = run_exp2(all_dfs_rl, count_index, utr3_seqs)
    csv2 = os.path.join(RESULTS_DIR, "exp2_off_target_analysis_RLv3.csv")
    exp2_df.to_csv(csv2, index=False)
    print(f"\nExp2 CSV → {csv2}", flush=True)

    exp2_stats = print_exp2_stats(exp2_df)
    plot_exp2_histogram(exp2_df, os.path.join(PLOTS_DIR, "exp2_off_target_histogram_RLv3.pdf"))
    plot_exp2_specificity(exp2_df, os.path.join(PLOTS_DIR, "exp2_specificity_comparison_RLv3.pdf"))

    # ── STEP 3: Experiment 3 — Potency ────────────────────────────────────────
    print("\n" + "=" * 70, flush=True)
    print("[Step 3] Experiment 3 — Functional Potency", flush=True)
    print("=" * 70, flush=True)

    # Part A — Context++ (all 3 lengths)
    exp3_df = run_exp3_part_a(all_dfs_rl)
    csv3 = os.path.join(RESULTS_DIR, "exp3_potency_summary_RLv3.csv")
    exp3_df.to_csv(csv3, index=False)
    print(f"\nExp3 Part A CSV → {csv3}", flush=True)

    exp3_stats = print_exp3_stats(exp3_df)
    plot_exp3_seed_potency(exp3_df, os.path.join(PLOTS_DIR, "exp3_seed_type_potency_RLv3.pdf"))
    plot_exp3_lfc(exp3_df, os.path.join(PLOTS_DIR, "exp3_lfc_distribution_RLv3.pdf"))

    # Part B — Cross-attention (100nt)
    print("\n[Exp3B] Cross-attention extraction (100nt, RL model) ...", flush=True)
    df_100 = all_dfs_rl["100nt"]
    seed_attn = run_exp3_part_b(model, df_100, device)

    exp3_sub  = exp3_df[exp3_df["mRNA_length"] == "100nt"].reset_index(drop=True)
    attn_conc = seed_attn.max(axis=1)   # peak attention over mRNA positions
    r, pval, rho, pval_sp = plot_exp3_attn_potency(
        attn_conc, exp3_sub["gen_context_score"].values,
        os.path.join(PLOTS_DIR, "exp3_attention_vs_potency_RLv3.pdf"),
    )
    print(f"  Cross-attn vs potency: Pearson r={r:.3f} p={pval:.3e}, "
          f"Spearman ρ={rho:.3f} p={pval_sp:.3e}", flush=True)

    # Save cross-attention CSV
    attn_df = pd.DataFrame({
        "Transcript_ID":        df_100["Transcript ID"],
        "miRNA_ID":             df_100["miRNA ID"],
        "seed_attn_peak":       attn_conc,
        "gen_context_score":    exp3_sub["gen_context_score"].values,
        "gen_seed_type":        exp3_sub["gen_seed_type"].values,
    })
    attn_csv = os.path.join(RESULTS_DIR, "exp3_cross_attention_100nt_RLv3.csv")
    attn_df.to_csv(attn_csv, index=False)
    print(f"  Cross-attn CSV → {attn_csv}", flush=True)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70, flush=True)
    print("ALL EXPERIMENTS COMPLETE (RL-finetuned checkpoint)", flush=True)
    print("=" * 70, flush=True)
    print(f"  Exp1 CSV:  {csv1}", flush=True)
    print(f"  Exp2 CSV:  {csv2}", flush=True)
    print(f"  Exp3 CSV:  {csv3}", flush=True)
    print(f"  Plots:     {PLOTS_DIR}/exp*_*_RLv3.pdf", flush=True)

    # Save combined stats as JSON for markdown update
    all_stats = {"exp1": exp1_stats, "exp2": exp2_stats, "exp3": exp3_stats,
                 "exp3_attn": {"pearson_r": float(r), "pearson_p": float(pval),
                               "spearman_rho": float(rho), "spearman_p": float(pval_sp)}}
    stats_json = os.path.join(RESULTS_DIR, "rl_experiment_stats_v3.json")
    with open(stats_json, "w") as fh:
        json.dump(all_stats, fh, indent=2)
    print(f"  Stats JSON → {stats_json}", flush=True)


if __name__ == "__main__":
    main()
