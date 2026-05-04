import os
import sys
import glob
import shutil
import argparse
import torch
import pandas as pd
from single_base_perturbation import predict, encode_seq, single_base_perturbation, viz_sequence
from plot_transformer_heatmap import plot_heatmap
from Data_pipeline import CharacterTokenizer
import DTEA_model as dtea
from Global_parameters import PROJ_HOME

data_dir = os.path.join(PROJ_HOME, "TargetScan_dataset")

def load_model(ckpt_path,
               **args_dict):
    # load model checkpoint
    model = dtea.DTEA(**args_dict)
    loaded_data = torch.load(ckpt_path, map_location=model.device)
    model.load_state_dict(loaded_data, strict=False)
    print(f"Loaded checkpoint from {ckpt_path}", flush=True)
    return model

def run_perturb(checkpoint_file, file_name, save_plot_dir=None):
    mirna_max_len   = 24
    mrna_max_len    = 520
    predict_span    = True
    predict_binding = True
    if torch.cuda.is_available():
        device = "cuda:2"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu" 
    args_dict = {"mirna_max_len": mirna_max_len,
                "mrna_max_len": mrna_max_len,
                "device": device,
                "embed_dim": 1024,
                "ff_dim": 4096,
                "num_heads": 8,
                "num_layers": 4,
                "predict_span": predict_span,
                "predict_binding": predict_binding,
                "use_longformer": True}
    print("Loading model ... ")
    model = load_model(checkpoint_file, **args_dict)
    
    test_data_path = os.path.join(data_dir, "TargetScan_train_500_randomized_start.csv")
    test_data = pd.read_csv(test_data_path)
    mRNA_seqs = test_data[["mRNA sequence"]].values
    miRNA_seqs = test_data[["miRNA sequence"]].values

    # === 关键改动：默认保存到 all_checkpoints_perturbation ===
    if save_plot_dir is None:
        save_plot_dir = os.path.join(
            PROJ_HOME,
            "Performance",
            "TargetScan_test",
            "viz_seq_perturbation",
            "TwoTowerTransformer",
            "500",
            "LSE+MIL",
        )
    os.makedirs(save_plot_dir, exist_ok=True)

    # Test sequence
    i=8
    mRNA_seq = mRNA_seqs[i][0]
    miRNA_seq = miRNA_seqs[i][0]
    miRNA_id = test_data[["miRNA ID"]].iloc[i,0]
    mRNA_id = test_data[["Transcript ID"]].iloc[i,0]
    seed_start = test_data[["seed start"]].iloc[i,0]
    seed_end = test_data[["seed end"]].iloc[i,0]
    print(f"miRNA id = {miRNA_id}", flush=True)
    print(f"mRNA id = {mRNA_id}", flush=True)

    # replace U with T and reverse miRNA to 3' to 5' 
    miRNA_seq = miRNA_seq.replace("U", "T")[::-1]
    
    # create tokenizer
    tokenizer = CharacterTokenizer(
        characters=["A", "T", "C", "G", "N"],  # add RNA characters, N is uncertain
        model_max_length=model.mrna_max_len,
        add_special_tokens=False,  # we handle special tokens elsewhere
        padding_side="right",  # since HyenaDNA is causal, we pad on the left
    )
    encoded = encode_seq(
        model=model, 
        tokenizer=tokenizer,
        mRNA_seq=mRNA_seq,
        miRNA_seq=miRNA_seq,
    )

    print("WT prediction ...", flush=True)
    model.to(args_dict["device"])
    wt_attn_score, wt_binding_prob, wt_binding_weights = predict(model, **encoded)
    
    # convert mRNA seq tokens to ids
    mRNA_tokens_squeezed = encoded["mRNA_seq"].squeeze(0)
    mRNA_ids = [tokenizer._convert_id_to_token(d.item()) for d in mRNA_tokens_squeezed]
    # convert miRNA seq tokens to ids
    miRNA_tokens_squeezed = encoded["miRNA_seq"].squeeze(0)
    miRNA_ids = [tokenizer._convert_id_to_token(d.item()) for d in miRNA_tokens_squeezed]

    fig, ax_attn = plot_heatmap(
        model,
        miRNA_seq=miRNA_ids,
        mRNA_seq=mRNA_ids,
        miRNA_id=miRNA_id,
        mRNA_id=mRNA_id,
        seed_start=seed_start,
        seed_end=seed_end,
        plot_max_only=True
    )

    # perturb mRNA sequence
    attn_deltas = []
    prob_deltas = []
    print("Start perturbing mrna ...")

    # Extract binding weights directly from model output (no perturbation needed)
    actual_mrna_len = len(mRNA_seq)
    if wt_binding_weights is not None:
        # wt_binding_weights has shape (batch_size, mrna_len), so we need to squeeze and slice correctly
        binding_weights_actual = wt_binding_weights.squeeze(0)[:actual_mrna_len]  # Extract weights for actual sequence
        weights_deltas = binding_weights_actual.tolist()  # Convert to list for visualization
        print(f"  weights_deltas: {len(weights_deltas)} elements, type: {type(weights_deltas)}")
    else:
        weights_deltas = None

    for pos in range(len(mRNA_seq)):
        attn_score_delta_list = []
        binding_prob_delta_list = []
        # perturb the base 3 times and average the delta (目前 loop 次数=1，与原始一致)
        for _ in range(1):
            perturbed_mRNA = single_base_perturbation(seq=mRNA_seq, pos=pos)
            encoded = encode_seq(
                model=model,
                tokenizer=tokenizer,
                mRNA_seq=perturbed_mRNA,
                miRNA_seq=miRNA_seq,
            )
            attn_score,  binding_prob, _ = predict(model, **encoded)
            attn_score_delta = abs(wt_attn_score - attn_score) # (L,)
            binding_prob_delta = abs(wt_binding_prob - binding_prob) 
            attn_score_delta_list.append(attn_score_delta)
            binding_prob_delta_list.append(binding_prob_delta)
        # only take the change at the current position
        all_attn_score_delta = torch.stack(attn_score_delta_list, dim=0)[:, pos] # (N, L) -> (N,)
        all_binding_prob_delta = torch.stack(binding_prob_delta_list) # (N, )
        attn_delta = (all_attn_score_delta.sum(dim=0) / len(all_attn_score_delta)).item()
        prob_delta = (all_binding_prob_delta.sum(dim=0) / len(all_binding_prob_delta)).item()
        attn_deltas.append(attn_delta)
        prob_deltas.append(prob_delta)
    print(f"Perturbation complete - {len(prob_deltas)} positions analyzed")
    
    # Debug: print final data types and shapes for visualization
    print(f"Final data for visualization:")
    print(f"  attn_deltas: {len(attn_deltas)} elements, type: {type(attn_deltas)}")
    print(f"  prob_deltas: {len(prob_deltas)} elements, type: {type(prob_deltas)}")
    print(f"  mRNA_seq length: {len(mRNA_seq)}")
    
    # print("Max in delta = ", max(deltas))
    print("plot changes on base logos ...", flush=True)
    file_path = os.path.join(save_plot_dir, file_name)
    fig, ax_viz = viz_sequence(
        seq=mRNA_seq,                 # visualize change on the original mRNA seq
        attn_changes=attn_deltas,
        weights_changes=weights_deltas,
        prob_changes=prob_deltas,
        seed_start=seed_start,
        seed_end=seed_end,
        base_ax=ax_attn,
        figsize=(45, 9),
        file_name=file_path
    )


def main():
    parser = argparse.ArgumentParser(
        description="Batch run run_perturb() for all .pth checkpoints in a folder."
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        required=True,
        help="Directory that contains all .pth checkpoint files.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="Starting index for file numbering (default: 1).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned actions without running run_perturb or moving files.",
    )
    args = parser.parse_args()

    ckpt_dir = os.path.abspath(args.ckpt_dir)
    if not os.path.isdir(ckpt_dir):
        print(f"[ERROR] ckpt-dir not found: {ckpt_dir}")
        sys.exit(1)

    # 收集并排序 .pth
    ckpt_list = sorted(glob.glob(os.path.join(ckpt_dir, "*.pth")))
    if not ckpt_list:
        print(f"[WARN] No .pth files found in: {ckpt_dir}")
        sys.exit(0)
    else:
        # generate file path for each checkpoint
        ckpt_list = [os.path.join(ckpt_dir, ckpt) for ckpt in ckpt_list]

    print(f"[INFO] Found {len(ckpt_list)} checkpoint(s) in {ckpt_dir}")

    # 逐个处理
    for idx, ckpt_path in enumerate(ckpt_list, start=args.start_index):
        ckpt_fname = os.path.basename(ckpt_path)
        ckpt_prefix = os.path.splitext(ckpt_fname)[0]
        out_png_name = f"{ckpt_prefix}_{str(idx)}.png"

        print(f"\n[INFO] Processing: {ckpt_fname}")
        print(f"       -> save name (inside run_perturb): {out_png_name}")

        if args.dry_run:
            # 仅打印动作
            print("[DRY-RUN] Would call run_perturb(...).")
            continue

        # 1) 调用你的函数：它会在内部目录里生成一个名为 out_png_name 的图
        try:
            run_perturb(ckpt_path, out_png_name)
        except Exception as e:
            print(f"[ERROR] run_perturb failed on {ckpt_fname}: {e}")
            continue

    print("\n[DONE] All checkpoints processed.")

if __name__ == "__main__":
    main()
