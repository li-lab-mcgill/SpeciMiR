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
import subprocess
import torch

from DTEA_model import CrossAttentionPredictor
from Data_pipeline import CharacterTokenizer
from Global_parameters import PROJ_HOME
from seed_match_verification import verify_anywhere_seed_matches
from Finetune_Specificity import load_discriminator_checkpoint

# from miRBench.encoder import get_encoder
# from miRBench.predictor import get_predictor

# ─── Configuration ────────────────────────────────────────────────────────────
MRNA_MAX_LEN = 50
MIRNA_MAX_LEN = 26 
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

RNACOFOLD_BIN = "/home/mcb/users/jgu13/.conda/envs/HyenaDNA_clone2/bin/RNAcofold"
RNAHYBRID_BIN = "/home/mcb/users/jgu13/.conda/envs/rnahybrid/bin/RNAhybrid"

N_SUBSAMPLE = 2000  # sufficient for statistical significance

class Discriminator_scorer():
    def __init__(self):
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
        self.disc_model = load_discriminator_checkpoint(disc_model, DISC_CKPT, DEVICE)
        self.tokenizer = CharacterTokenizer(characters=["A", "T", "C", "G", "N"],
                                       model_max_length=MRNA_MAX_LEN,
                                       padding_side="right")
        
    @torch.no_grad()
    def score_batch(self, batch, mirna_col, mrna_col):
        self.disc_model.to(DEVICE)
        self.disc_model.eval()
        mirna_seqs = batch[mirna_col]
        mrna_seqs = batch[mrna_col]
        mirna_ids, mrna_ids = [], []
        mirna_masks, mrna_masks = [], []
        n_tokenized = 0
        for mirna_seq, mrna_seq in zip(mirna_seqs, mrna_seqs):
            mirna_tok = self.tokenizer(mirna_seq, padding="max_length", truncation=True, max_length=MIRNA_MAX_LEN, return_attention_mask=True)
            mrna_tok = self.tokenizer(mrna_seq, padding="max_length", truncation=True, max_length=MRNA_MAX_LEN, return_attention_mask=True)
            mirna_ids.append(mirna_tok["input_ids"])
            mrna_ids.append(mrna_tok["input_ids"])
            mirna_masks.append(mirna_tok["attention_mask"])
            mrna_masks.append(mrna_tok["attention_mask"])
            n_tokenized += 1
            if n_tokenized % BATCH_SIZE == 0:
                print(f"Tokenized {n_tokenized} / {len(mirna_seqs)} sequences", flush=True)

        mirna_ids = torch.tensor(mirna_ids, dtype=torch.long).to(DEVICE)
        mrna_ids = torch.tensor(mrna_ids, dtype=torch.long).to(DEVICE)
        mirna_masks = torch.tensor(mirna_masks, dtype=torch.long).to(DEVICE)
        mrna_masks = torch.tensor(mrna_masks, dtype=torch.long).to(DEVICE)

        mrna_sn_embedding = self.disc_model.sn_embedding(mrna_ids)
        mrna_cnn_embedding = self.disc_model.cnn_embedding(mrna_sn_embedding.transpose(-1, -2))
        mrna_embedding = mrna_sn_embedding + mrna_cnn_embedding
        
        mirna_sn_embedding = self.disc_model.sn_embedding(mirna_ids)
        mirna_cnn_embedding = self.disc_model.cnn_embedding(mirna_sn_embedding.transpose(-1, -2))
        mirna_embedding = mirna_sn_embedding + mirna_cnn_embedding
            
        mirna_embedding = self.disc_model.mirna_encoder(mirna_embedding, mask=mirna_masks)
        mrna_embedding = self.disc_model.mrna_encoder(mrna_embedding, mask=mrna_masks)

        z = self.disc_model.cross_attn_layer(query=mrna_embedding, 
                                    key=mirna_embedding,
                                    value=mirna_embedding,
                                    mask=mirna_masks)
        z_res = self.disc_model.dropout(z) + mrna_embedding 
        z_norm = self.disc_model.cross_norm(z_res)
        z_norm = z_norm.masked_fill(mrna_masks.unsqueeze(-1)==0, 0) 

        valid_counts = mrna_masks.sum(dim=1, keepdim=True)
        pooled = z_norm.sum(dim=1) / (valid_counts + 1e-8)
        logit = self.disc_model.binding_output(pooled).squeeze(-1)   
        scores = torch.sigmoid(logit).detach().cpu().numpy()
        return scores

class RNACofold_scorer():
    def __init__(self):
        self.rnacofold_bin = RNACOFOLD_BIN
    
    def _score_single(self, mirna_seq, mrna_seq):
        mirna_seq = mirna_seq.replace("T", "U")
        mrna_seq = mrna_seq.replace("T", "U")
        input_line = "&".join([mirna_seq, mrna_seq])

        try:
            results = subprocess.run(
                # -a partition function including delta G binding, -d2 dangling end energy treatment, 
                # --noLP no long-range pairing
                [self.rnacofold_bin, "-a", "-d2"], 
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
    
    def score_batch(self, batch, mirna_col, mrna_col):
        mirna_seqs = batch[mirna_col]
        mrna_seqs = batch[mrna_col]
        scores = []
        n_done = 0
        for i, (mirna_seq, mrna_seq) in enumerate(zip(mirna_seqs, mrna_seqs)):
            scores.append(self._score_single(mirna_seq, mrna_seq))
            if i % BATCH_SIZE == 0:
                print(f"RNACofold scored {n_done} / {len(mirna_seqs)} sequences", flush=True)
                n_done += BATCH_SIZE
        return np.asarray(scores)

class RNAHybrid_scorer():
    def __init__(self):
        self.rnahybrid_bin = RNAHYBRID_BIN
    
    def _score_single(self, mirna_seq, mrna_seq):
        mirna_seq = mirna_seq.replace("T", "U")
        mrna_seq = mrna_seq.replace("T", "U")

        try:
            results = subprocess.run(
                [self.rnahybrid_bin, 
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
    
    def score_batch(self, batch, mirna_col, mrna_col):
        mirna_seqs = batch[mirna_col]
        mrna_seqs = batch[mrna_col]
        scores = []
        n_done = 0
        for i, (mirna_seq, mrna_seq) in enumerate(zip(mirna_seqs, mrna_seqs)):
            scores.append(self._score_single(mirna_seq, mrna_seq))
            if i % BATCH_SIZE == 0:
                print(f"RNAhybrid scored {n_done} / {len(mirna_seqs)} sequences", flush=True)
                n_done += BATCH_SIZE
        return np.asarray(scores)

# class miRBind_scorer():
#     def __init__(self):
#         self.mirbind_encoder = get_encoder("miRBind_Klimentova2022")
#         self.mirbind_decoder = get_predictor("miRBind_Klimentova2022")
#         self.mirbind_decoder.to(DEVICE)
    
#     def score_batch(self, batch, mirna_col, mrna_col):
#         embeddings = self.mirbind_encoder(batch, miRNA_col=mirna_col, gene_col=mrna_col)
#         embeddings = embeddings.to(DEVICE)
#         scores = self.mirbind_decoder(embeddings)
#         return scores.detach().cpu().numpy()

def make_random_mirna(target, mirna_len=22):
    """
    Create a random miRNA that does not contain any seed matches to the target mRNA.
    """
    match_seed = True
    while match_seed:
        rand_mirna = "".join(random.choices(BASES, k=mirna_len))
        found, _, _, _ = verify_anywhere_seed_matches(target, rand_mirna)
        if not found:
            match_seed = False
    return rand_mirna

def main():
    # ─── Load data ────────────────────────────────────────────────────────────────
    datapath = os.path.join(PROJ_HOME, "Manakov2022/AGO2_eCLIP_Manakov2022_test_predicted.tsv.gz")
    dataset = pd.read_csv(datapath, compression="gzip", sep="\t")
    dataset = dataset.sample(n=N_SUBSAMPLE, random_state=SEED, ignore_index=True)
    all_mrnas = dataset["gene"].unique().tolist()
    # filter for only target mRNAs
    dataset = dataset[dataset["label"] == 1]
    # filter for only generated miRNA sequences
    dataset = dataset[dataset["generated_mirna"] != "NA"]
    # filter for only real miRNA sequences
    dataset = dataset[dataset["noncodingRNA"] != "NA"]
    # build real mirna to target mrna mapping
    real_mirna_to_mrna = dataset[["gene", "noncodingRNA"]]
    real_mirna_to_mrna.columns = ["gene", "mirna"]

    # build generated mirna to target mrna mapping
    gen_mirna_to_mrna = dataset[["gene", "generated_mirna"]]
    gen_mirna_to_mrna.columns = ["gene", "mirna"]

    # # filter for `K_OFF` off-target mRNAs for each generated miRNA
    # gen_off_mirna = []
    # gen_off_mrna = []
    # for _, row in dataset.iterrows():
    #     target = row["gene"]
    #     gen_seq = row["generated_mirna"]
    #     # Sample k_off off-targets (excluding the target itself)
    #     candidates = [m for m in random.sample(all_mrnas, min(K_OFF * 2, len(all_mrnas)))
    #                    if m != target][:K_OFF]
    #     while len(candidates) < K_OFF:
    #         candidates.append(random.choice(all_mrnas))
    #     for off_mrna in candidates:
    #         gen_off_mirna.append(gen_seq)
    #         gen_off_mrna.append(off_mrna)
    # gen_off_dataset = pd.DataFrame({"gene": gen_off_mrna, "mirna": gen_off_mirna})
    # print(f"Generated off-target dataset head: {gen_off_dataset.head()}")
    # # save generated off-target dataset
    # gen_off_dataset.to_csv(os.path.join(PROJ_HOME, "Manakov2022/AGO2_eCLIP_Manakov2022_test_predicted_generated_off_target.tsv"), sep="\t", index=False)
    # print("Generated off-target dataset saved to Manakov2022/AGO2_eCLIP_Manakov2022_test_predicted_generated_off_target.tsv")

    # # create random miRNA (not seed-matched) for each target mRNA
    # random_mirnas = []
    # target_mrnas = all_mrnas
    # for target in all_mrnas:    
    #     rand_mirna = make_random_mirna(target)
    #     random_mirnas.append(rand_mirna)
    # random_dataset = pd.DataFrame({"gene": target_mrnas, "mirna": random_mirnas})
    # print(f"Random dataset head: {random_dataset.head()}")
    # # save random dataset
    # random_dataset.to_csv(os.path.join(PROJ_HOME, "Manakov2022/AGO2_eCLIP_Manakov2022_test_predicted_random.tsv"), sep="\t", index=False)
    # print("Random dataset saved to Manakov2022/AGO2_eCLIP_Manakov2022_test_predicted_random.tsv")

    # load generated off-target dataset
    gen_off_dataset = pd.read_csv(os.path.join(PROJ_HOME, "Manakov2022/AGO2_eCLIP_Manakov2022_test_predicted_generated_off_target.tsv.gz"), compression="gzip", sep="\t")
    # load random dataset
    random_dataset = pd.read_csv(os.path.join(PROJ_HOME, "Manakov2022/AGO2_eCLIP_Manakov2022_test_predicted_random.tsv.gz"), compression="gzip", sep="\t")
    
    datasets = {
        "real_miRNA_target_mRNA": real_mirna_to_mrna,
        "generated_miRNA_target_mRNA": gen_mirna_to_mrna,
        "generated_miRNA_off_target_mRNA": gen_off_dataset,
        "random_miRNA_target_mRNA": random_dataset,
    }
    discriminator_scorer = Discriminator_scorer()
    rnacofold_scorer = RNACofold_scorer()
    rnahybrid_scorer = RNAHybrid_scorer()
    # mirbind_scorer = miRBind_scorer()
    
    scorers = {"Discriminator": discriminator_scorer, "RNACofold": rnacofold_scorer, "RNAHybrid": rnahybrid_scorer}#, "miRBind": mirbind_scorer}

    # ─── Score datasets ────────────────────────────────────────────────────────────
    results = {dataset_name: {scorer_name: [] for scorer_name in scorers.keys()} for dataset_name in datasets.keys()}
    for dataset_name, dataset in datasets.items():
        for scorer_name, scorer in scorers.items():
            scores = scorer.score_batch(dataset, mirna_col="mirna", mrna_col="gene")
            results[dataset_name][scorer_name].extend(scores)
            # Save scores for this dataset and scorer immediately
            scores_df = pd.DataFrame({
                "gene": dataset["gene"],
                "mirna": dataset["mirna"],
                "score": scores
            })
            # After saving the scores_df for each scorer and dataset, compute and print mean and std immediately
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            print(f"  {dataset_name} / {scorer_name}: mean = {mean_score:.4f}, std = {std_score:.4f}")
            # Save per scorer, per dataset, tsv for easier external analysis (append or write, or overwrite)
            out_filename = f"Table2Three_scorers_scores_{dataset_name}_{scorer_name}.tsv.gz"
            out_path = os.path.join(PROJ_HOME, os.path.dirname(datapath), out_filename)
            scores_df.to_csv(out_path, sep="\t", index=False, compression="gzip")
            print(f"Saved scores for {dataset_name} / {scorer_name} to {out_path}")
   
    # ─── Save results ────────────────────────────────────────────────────────────
    # Calculate and save the mean and std of each result set
    summary_rows = []
    for dataset_name in results:
        for scorer_name in results[dataset_name]:
            scores = np.array(results[dataset_name][scorer_name])
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            summary_rows.append({
                "dataset": dataset_name,
                "method": scorer_name,
                "mean": mean_score,
                "std": std_score
            })

    summary_df = pd.DataFrame(summary_rows)
    # Save to CSV
    summary_df.to_csv(os.path.join(PROJ_HOME, os.path.dirname(datapath), "Table2Three_scorers_summary.tsv"), index=False, sep="\t")
    print(f"Summary statistics saved to {os.path.join(PROJ_HOME, os.path.dirname(datapath), 'Table2Three_scorers_summary.tsv')}")

if __name__ == "__main__":
    main()