"""
This script loads the siRNA targets from the FDA siRNA data and then feed it to 
the Manakov2022 trained generator to generate miRNAs. Then evaluate the generated
miRNAs by:
1. direct comparison with the ground truth siRNA
2. compare the seed region of the generated miRNAs with the ground truth siRNA
3. Thermodynamics
"""
import os
import pandas as pd
import subprocess
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from Data_pipeline import TargetPredictionDataset, CharacterTokenizer

from Bio import Align
from Bio.pairwise2 import align as pw2_align, format_alignment

from Global_parameters import PROJ_HOME, RNACOFOLD_BIN, RNAHYBRID_BIN
from DTEA_model import TargetGenerationModel
from Table2Three_scorers import Discriminator_scorer, RNACofold_scorer, RNAHybrid_scorer

def reverse_complement(seq):
    # reverse the sequence and complement the base
    comp = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join(comp.get(b) for b in reversed(seq))

def check_complementarity(generated_mirna, target_mRNA):
    # check whether the generated mirna is complementary to the target mRNA
    rc_mirna = reverse_complement(generated_mirna)
    return target_mRNA.find(rc_mirna) != -1

def align_mirna_to_mrna(mrna_5to3, mirna_3to5):
    """
    miRNA 3'→5' pairs antiparallel with mRNA 5'→3'.
    
    Steps:
      1. Reverse the miRNA → now 5'→3' (same direction as mRNA)
      2. Complement it → now represents what the mRNA SHOULD look like at binding site
      3. Smith-Waterman align this against the actual mRNA (identity-based)
      4. Map alignment back to original miRNA orientation for display
    """
    # miRNA 3'→5' → reverse → 5'→3' → complement → expected mRNA 5'→3'
    mirna_5to3 = mirna_3to5[::-1]
    mirna_as_mrna = reverse_complement(mirna_5to3) 

    # Smith-Waterman local alignment
    aligner = Align.PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -0.5

    alignments = aligner.align(mrna_5to3, mirna_as_mrna)
    best = alignments[0]

    return best, mirna_as_mrna

def semi_global_align(mrna_5to3, mirna_3to5):
    """
    Semi-global: global on miRNA, local on mRNA.
    The entire miRNA is aligned, but only the best-fitting
    region of the mRNA is used.
    """
    # Complement miRNA 3'→5' to get expected mRNA 5'→3'
    comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    mirna_as_mrna = "".join(comp[b] for b in mirna_3to5)

    aligner = Align.PairwiseAligner()
    aligner.mode = "global"  # global mode, but we control gaps below

    aligner.match_score = 2
    aligner.mismatch_score = -1

    # Internal gaps (bulges in duplex): penalized
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -0.5

    # mRNA end gaps: FREE (local on mRNA)
    aligner.target_left_open_gap_score = 0
    aligner.target_left_extend_gap_score = 0
    aligner.target_right_open_gap_score = 0
    aligner.target_right_extend_gap_score = 0

    # miRNA end gaps: PENALIZED (global on miRNA)
    aligner.query_left_open_gap_score = -5
    aligner.query_left_extend_gap_score = -0.5
    aligner.query_right_open_gap_score = -5
    aligner.query_right_extend_gap_score = -0.5

    # target = mRNA, query = mirna_as_mrna
    alignments = aligner.align(mrna_5to3, mirna_as_mrna)
    best = alignments[0]

    return best, mirna_as_mrna

def display_biology_alignment(mrna_5to3, mirna_3to5):
    """
    Run Smith-Waterman, then display in standard miRNA biology format:
    
    mRNA  5' ---NNNNNNNN--- 3'
                ||||||||
    miRNA 3' ---NNNNNNNN--- 5'
    """
    # best, mirna_as_mrna = align_mirna_to_mrna(mrna_5to3, mirna_3to5)
    best, mirna_as_mrna = semi_global_align(mrna_5to3, mirna_3to5)

    # Extract aligned regions from the alignment object
    mrna_aln = ""
    mirna_mrna_aln = ""
    
    # Get alignment coordinates
    aligned = best.aligned
    # aligned[0] = target (mRNA) coordinates
    # aligned[1] = query (mirna_as_mrna) coordinates
    
    target_start = aligned[0][0][0]
    target_end = aligned[0][-1][1]
    query_start = aligned[1][0][0]
    query_end = aligned[1][-1][1]

    # Get the formatted alignment string
    aln_str = str(best)
    lines = aln_str.strip().split("\n")
    # BioPython format: line 0 = target, line 1 = match, line 2 = query
    mrna_aligned = lines[0].strip().split()[-1] if lines else ""
    query_aligned = lines[2].strip().split()[-1] if len(lines) > 2 else ""

    # Map back: mirna_as_mrna → original miRNA orientation
    # mirna_as_mrna[i] is complement of mirna_3to5[i]
    # So the aligned region of mirna_as_mrna corresponds to the same
    # positions in mirna_3to5

    # Build display
    comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N", "-": "-"}
    
    # Use the raw alignment format
    print(f"Smith-Waterman alignment score: {best.score}")
    print(f"mRNA binding region: [{target_start}:{target_end}]")
    print(f"miRNA positions used: [{query_start}:{query_end}]")
    print()

    # Build character-by-character alignment display
    mrna_slice = mrna_5to3[target_start:target_end]
    mirna_slice = mirna_3to5[query_start:query_end]
    
    # Re-do pairwise for clean character-level output
    alns = pw2_align.localms(
        mrna_slice, 
        "".join(comp.get(b, "N") for b in mirna_slice),  # complement of miRNA 3'→5' slice
        2, -1, -5, -0.5,
        one_alignment_only=True,
    )
    
    if not alns:
        print("No alignment found")
        return
        
    aln = alns[0]
    mrna_a, query_a = aln.seqA, aln.seqB
    
    # Convert query back to miRNA bases (un-complement)
    mirna_display = ""
    match_line = ""
    for m, q in zip(mrna_a, query_a):
        if q == "-":
            mirna_display += "-"
            match_line += " "
        else:
            mi_base = comp.get(q, "N")  # un-complement to get original miRNA base
            mirna_display += mi_base
            if m == q:  # q is complement of mi_base, m matches q → Watson-Crick pair
                match_line += "|"
            elif (m == "G" and mi_base == "T") or (m == "T" and mi_base == "G"):
                match_line += ":"  # G:U wobble
            elif m == "-":
                match_line += " "
            else:
                match_line += " "

    # Flanking context
    flank = 3
    left = mrna_5to3[max(0, target_start-flank):target_start]
    right = mrna_5to3[target_end:min(len(mrna_5to3), target_end+flank)]
    pad = " " * len(left)

    print(f"mRNA  5'  ...{left}{mrna_a}{right}...  3'")
    print(f"              {pad}{match_line}")
    print(f"miRNA 3'     {pad}{mirna_display}        5'")

    wc = match_line.count("|")
    wobble = match_line.count(":")
    total = len(mirna_slice)
    print(f"\nWatson-Crick: {wc}/{total}, Wobble: {wobble}, "
          f"Mismatch: {total - wc - wobble}")
    
    return target_start, target_end, match_line

def RNACofold_score(mirna_seq, mrna_seq, rnacofold_bin=RNACOFOLD_BIN):
    """
    Score the duplex stability of the miRNA-mRNA using RNACofold
    """
    mirna_seq = mirna_seq.replace("T", "U")
    mrna_seq = mrna_seq.replace("T", "U")
    input_line = "&".join([mirna_seq, mrna_seq])

    try:
        results = subprocess.run(
            # -a partition function including delta G binding, -d2 dangling end energy treatment, 
            # --noLP no long-range pairing
            [rnacofold_bin, "-a", "-d2"], 
            input=input_line,
            capture_output=True,
            text=True,
            timeout=30
        )
        if results.returncode != 0:
            print(f"RNACofold failed: {results.stderr}", flush=True)
            return np.nan

        output_lines = [l for l in results.stdout.splitlines() if l.strip()]
        for line in output_lines:
            if "delta G binding=" in line:
                # "... delta G binding= -5..."
                delta_g_binding = float(line.split("delta G binding=")[1].strip().split(" ")[0])
                return delta_g_binding
        
        print(f"RNACofold output missing delta G binding: {results.stdout}", flush=True)
        return np.nan

    except Exception as e:
        print(f"Error running RNACofold: {e}", flush=True)
        return np.nan
    
def RNAHybrid_score(mirna_seq, mrna_seq, rnahybrid_bin=RNAHYBRID_BIN):
    """
    Score the change in free energy of the miRNA-mRNA duplex using RNAhybrid
    """
    mirna_seq = mirna_seq.replace("T", "U")
    mrna_seq = mrna_seq.replace("T", "U")

    try:
        results = subprocess.run(
            [rnahybrid_bin, 
            "-c",       # compact output
            "-b", "1",  # 1 hit per target
            "-d", "1.28, 0.9", # manually provide xi and theta parameters for calculating P-value of MFE, will not affect the MFE score
            mrna_seq,   # target (long mRNA)
            mirna_seq,  # query  (short miRNA)
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        mfe_str = results.stdout.split(":")[4] # "command_line:100:command_line:22:-21.4..."
        mfe = float(mfe_str)
        return mfe
    except Exception as e:
        print(f"Error running RNAhybrid: {e}", flush=True)
        return np.nan

def find_seed_matched_off_targets(target_seed_region, test_mrnas_df, target_clusters, max_samples=2000):
    """
    Find the off-targets that contain the target seed region
    """
    found_off_targets = []
    for i, row in test_mrnas_df.iterrows():
        mrna_seq = row['gene']
        if row['gene_cluster_ID'] not in target_clusters and target_seed_region in mrna_seq:
            found_off_targets.append(mrna_seq)
        if len(found_off_targets) >= max_samples: # stop after finding max_samples off-targets
            break
    return found_off_targets

# ── Paths ──────────────────────────────────────────────────────────────────────
si_rna_targets_path = os.path.join(PROJ_HOME, "siRNA_targets_data", "fda_sirna_targets.csv")
test_mrnas_path = os.path.join(PROJ_HOME, "Manakov2022", "AGO2_eCLIP_Manakov2022_test.tsv.gz")
ckpt_path = os.path.join(PROJ_HOME, "checkpoints/specificity_gen/Manakov2022_train/best_loss_0.5408_epoch8.pth")

# ── Load the model ────────────────────────────────────────────────────────────────
# MRNA_MAX_LEN = 80
# MIRNA_MAX_LEN = 26 # 24 + 2
# BATCH_SIZE = 64
# DEVICE = "cuda:4"
# SEED = 42
# EMBED_DIM = 1024
# NUM_HEADS = 8
# NUM_LAYERS = 4
# FF_DIM = 4096
# USE_LONGFORMER = True

# si_rna_targets_df = pd.read_csv(si_rna_targets_path, sep='\t')
# tokenizer = CharacterTokenizer(characters=["A", "T", "C", "G", "N"],
#                 add_special_tokens=False, 
#                 model_max_length=MRNA_MAX_LEN-2, # minus 2 for BOS and EOS tokens
#                 padding_side="right")
# ds_test = TargetPredictionDataset(data=si_rna_targets_df,
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

# # ── Generate miRNA sequences ──────────────────────────────────────────────────────
# print(f"Generating miRNA sequences for {len(si_rna_targets_df)} target mRNAs...")
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
#     print(f"  Generated {n_done}/{len(si_rna_targets_df)} sequences", end="\r", flush=True)
# print()
# si_rna_targets_df["generated_mirna"] = all_generated_seqs
# si_rna_targets_df.to_csv(si_rna_targets_path, index=False, sep='\t')
# print(f"Generated mirna saved to {si_rna_targets_path}")

# ── Evaluate the generated miRNAs ──────────────────────────────────────────────────
print(f"Evaluating the generated miRNAs...")
# load generated mirnas and target mrnas from sirna_targets_path
sirna_targets_df = pd.read_csv(si_rna_targets_path, sep='\t')
test_mrnas_df = pd.read_csv(test_mrnas_path, sep='\t', compression='gzip')
off_target_path = os.path.join(PROJ_HOME, "siRNA_targets_data", "off_targets.tsv.gz")
off_target_df = pd.read_csv(off_target_path, sep='\t', compression='gzip')

# three scorers
discriminator_scorer = Discriminator_scorer()
rnacofold_scorer = RNACofold_scorer()
rnahybrid_scorer = RNAHybrid_scorer()
# save summary statistics
summary_path = os.path.join(PROJ_HOME, "siRNA_targets_data", "off_target_specificity_summary.tsv")
summary_rows = []

for i, row in sirna_targets_df.iterrows():
    drug_name = row["drug"]
    generated_mirna_3to5 = row["generated_mirna"]
    ground_truth_sirna = row["noncodingRNA"]
    generated_mirna_5to3 = generated_mirna_3to5[::-1]
    fda_sirna = row["noncodingRNA"]
    target_mRNA = row["gene"]
    on_target_mrnas = {
        "noncodingRNA": [fda_sirna],
        "generated_mirna": [generated_mirna_5to3],
        "gene": [target_mRNA],
    }
    on_target_mrnas_df = pd.DataFrame(on_target_mrnas)
    print(f"--------Drug name: {drug_name}--------")
    # off_targets = off_target_df[off_target_df["drug_name"] == drug_name].sample(n=1000, random_state=42)
    # -----Check generated mirna is complementary to the target mRNA---
    display_biology_alignment(mrna_5to3=target_mRNA, mirna_3to5=generated_mirna_3to5)
    # -----Check thermodynamics of the generated miRNA and target mRNA-----
    # rnacofold_score = RNACofold_score(mirna_seq=generated_mirna_3to5, mrna_seq=target_mRNA)
    # rnacofold_score_fda = RNACofold_score(mirna_seq=fda_sirna, mrna_seq=target_mRNA)
    # rnahybrid_score = RNAHybrid_score(mirna_seq=generated_mirna_3to5, mrna_seq=target_mRNA)
    # rnahybrid_score_fda = RNAHybrid_score(mirna_seq=fda_sirna, mrna_seq=target_mRNA)
    # check the percentage of energy of the generated miRNA to the ground truth siRNA
    # rnacofold_percentage_energy_change = (rnacofold_score / rnacofold_score_fda) * 100
    # rnahybrid_percentage_energy_change = (rnahybrid_score / rnahybrid_score_fda) * 100
    # print(f"RNACofold score (generated): {rnacofold_score}")
    # print(f"RNAHybrid score (generated): {rnahybrid_score}")
    # print(f"Percentage of energy change (generated to FDA): {rnacofold_percentage_energy_change}%")
    # print(f"RNACofold score (FDA): {rnacofold_score_fda}")
    # print(f"RNAHybrid score (FDA): {rnahybrid_score_fda}")
    # print(f"Percentage of energy change (generated to FDA): {rnahybrid_percentage_energy_change}%")
    # ------Check the specificity of the siRNA and the generated miRNA----
    # score with trained discriminator
    # discriminator_scores = discriminator_scorer.score_batch(batch=on_target_mrnas_df, mirna_col="generated_mirna", mrna_col="gene")
    # discriminator_scores_fda = discriminator_scorer.score_batch(batch=on_target_mrnas_df, mirna_col="noncodingRNA", mrna_col="gene")
    # print(f"Discriminator score (generated): {discriminator_scores}")
    # print(f"Discriminator score (FDA): {discriminator_scores_fda}")
    # score with RNACofold
    # rnacofold_scores = rnacofold_scorer.score_batch(batch=off_targets, mirna_col="generated_mirna", mrna_col="off_target_gene")
    # rnacofold_scores_fda = rnacofold_scorer.score_batch(batch=off_targets, mirna_col="noncodingRNA", mrna_col="off_target_gene")
    # score with RNAHybrid
    # rnahybrid_scores = rnahybrid_scorer.score_batch(batch=off_targets, mirna_col="generated_mirna", mrna_col="off_target_gene")
    # rnahybrid_scores_fda = rnahybrid_scorer.score_batch(batch=off_targets, mirna_col="noncodingRNA", mrna_col="off_target_gene")

    # Save scores for this dataset and scorer immediately
    # scores_df = pd.DataFrame({
    #     "gene": off_targets["off_target_gene"],
    #     "generated_mirna": off_targets["generated_mirna"],
    #     "noncodingRNA": off_targets["noncodingRNA"],
    #     "discriminator_score": discriminator_scores,
    #     "discriminator_score_fda": discriminator_scores_fda,
    #     "rnacofold_score": rnacofold_scores,
    #     "rnacofold_score_fda": rnacofold_scores_fda,
    #     "rnahybrid_score": rnahybrid_scores,
    #     "rnahybrid_score_fda": rnahybrid_scores_fda
    # })
    # After saving the scores_df for each scorer and dataset, compute and print mean and std immediately
    # mean_discriminator_score = np.mean(discriminator_scores)
    # mead_discriminator_score_fda = np.mean(discriminator_scores_fda)
    # std_discriminator_score = np.std(discriminator_scores)
    # std_discriminator_score_fda = np.std(discriminator_scores_fda)
    # print(f"  Discriminator score: mean = {mean_discriminator_score:.4f}, std = {std_discriminator_score:.4f}")
    # print(f"  Discriminator score (FDA): mean = {mead_discriminator_score_fda:.4f}, std = {std_discriminator_score_fda:.4f}")
    # mean_rnacofold_score = np.mean(rnacofold_scores)
    # mean_rnacofold_score_fda = np.mean(rnacofold_scores_fda)
    # std_rnacofold_score = np.std(rnacofold_scores)
    # std_rnacofold_score_fda = np.std(rnacofold_scores_fda)
    # print(f"  RNACofold score: mean = {mean_rnacofold_score:.4f}, std = {std_rnacofold_score:.4f}")
    # print(f"  RNACofold score (FDA): mean = {mean_rnacofold_score_fda:.4f}, std = {std_rnacofold_score_fda:.4f}")
    # mean_rnahybrid_score = np.mean(rnahybrid_scores)
    # mean_rnahybrid_score_fda = np.mean(rnahybrid_scores_fda)
    # std_rnahybrid_score = np.std(rnahybrid_scores)
    # std_rnahybrid_score_fda = np.std(rnahybrid_scores_fda)
    # print(f"  RNAHybrid score: mean = {mean_rnahybrid_score:.4f}, std = {std_rnahybrid_score:.4f}")
    # print(f"  RNAHybrid score (FDA): mean = {mean_rnahybrid_score_fda:.4f}, std = {std_rnahybrid_score_fda:.4f}")

#     summary_rows.append({
#         "drug_name": drug_name,
#         "method": "discriminator",
#         "mean": mean_discriminator_score,
#         "std": std_discriminator_score,
#         "mean_fda": mead_discriminator_score_fda,
#         "std_fda": std_discriminator_score_fda
#     })
#     summary_rows.append({
#         "drug_name": drug_name,
#         "method": "rnacofold",
#         "mean": mean_rnacofold_score,
#         "std": std_rnacofold_score,
#         "mean_fda": mean_rnacofold_score_fda,
#         "std_fda": std_rnacofold_score_fda
#     })
#     summary_rows.append({
#         "drug_name": drug_name,
#         "method": "rnahybrid",
#         "mean": mean_rnahybrid_score,
#         "std": std_rnahybrid_score,
#         "mean_fda": mean_rnahybrid_score_fda,
#         "std_fda": std_rnahybrid_score_fda
#     })

# summary_df = pd.DataFrame(summary_rows)
# # Save to CSV
# summary_df.to_csv(summary_path, index=False, sep="\t")
# print(f"Summary statistics saved to {summary_path}")