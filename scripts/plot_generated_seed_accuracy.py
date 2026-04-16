'''
This script plots the 
'''
import os
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind
from Global_parameters import PROJ_HOME, AXIS_FONT_SIZE, TICK_FONT_SIZE, TITLE_FONT_SIZE, LEGEND_FONT_SIZE
import mpmath as mp


SEED_START_COL = "seed start"
SEED_END_COL = "seed end"
MIRNA_SEQ_COL = "miRNA sequence"
GENERATED_SEQ_COL = "generated_mirna"


def _clean_sequence(value: str) -> str:
    """Upper-case the sequence and strip whitespace so comparisons are uniform."""
    # convert all U to T in mirna sequence
    value = value.replace("U", "T")
    return "".join(str(value).upper().split()) if isinstance(value, str) else ""


def compute_accuracy(df: pd.DataFrame) -> tuple[dict[str, float], list[float]]:
    total_matches = total_bases = 0
    seed_matches = seed_bases = 0
    non_seed_matches = non_seed_bases = 0
    per_seed_accuracy: list[float] = []

    for _, row in df.iterrows():
        ref_seq = _clean_sequence(row.get(MIRNA_SEQ_COL, ""))
        gen_seq = _clean_sequence(row.get(GENERATED_SEQ_COL, ""))
        gen_seq = gen_seq[::-1] # because the generated miRNA is reversed (3' to 5')
        if not ref_seq or not gen_seq:
            continue

        ref_len = len(ref_seq)
        gen_len = len(gen_seq)
        row_matches = 0

        for idx, base in enumerate(ref_seq):
            if idx < gen_len and base == gen_seq[idx]:
                row_matches += 1

        seed_start, seed_end = row.get(SEED_START_COL), row.get(SEED_END_COL)
        seed_len = seed_end - seed_start + 1
        row_seed_matches = 0

        mirna_start, mirna_end = 1, 1+seed_len
        
        if seed_len > 0:
            for idx in range(mirna_start, mirna_end):
                if idx < gen_len and ref_seq[idx] == gen_seq[idx]:
                    row_seed_matches += 1
            per_seed_accuracy.append(row_seed_matches / seed_len)
        else:
            per_seed_accuracy.append(math.nan)

        total_matches += row_matches
        total_bases += ref_len
        seed_matches += row_seed_matches
        seed_bases += seed_len

        non_seed_len = ref_len - seed_len
        non_seed_bases += non_seed_len
        non_seed_matches += row_matches - row_seed_matches

    metrics = {}
    if total_bases:
        metrics["Overall"] = total_matches / total_bases
    if seed_bases:
        metrics["Seed"] = seed_matches / seed_bases
    if non_seed_bases:
        metrics["Non-seed"] = non_seed_matches / non_seed_bases

    return metrics, [acc for acc in per_seed_accuracy if not math.isnan(acc)]


def plot_accuracy(metrics: list[dict[str, float]], model_names: list[str], colors: dict[str, str]) -> None:
    """
    Plot the accuracy of the predicted miRNA seed and non-seed.
    """
    categories = ["Overall", "Seed", "Non-seed"]
    n_models = len(metrics)
    fig, ax = plt.subplots(figsize=(2 * n_models, 3))
    width = 0.5 * (1/n_models)
    x = np.arange(len(categories))
    color_list = list(colors.values())
    for i, (metric, model_name, color) in enumerate(zip(metrics, model_names, color_list)):
        values = [metric.get(cat, 0.0) for cat in categories]
        bars = ax.bar(x + i * width, values, width, color=color, label=model_name)
        # Add value labels on top of each bar.
        for rect, val in zip(bars, values):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )
    
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Predicted miRNA seed vs non-seed accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(loc=(1.0, 0.8), fontsize=LEGEND_FONT_SIZE-2, frameon=False)
    save_path = os.path.join(PROJ_HOME, "Performance", "TargetScan_test", "TwoTowerTransformer", "generated_seed_and_non_seed_accuracy_mrna_length_comparison.svg")
    # save the figure to the project home
    fig.savefig(save_path, dpi=400, bbox_inches="tight")
    print(f"Figure saved to {save_path}")

def plot_per_seed_accuracy(per_seed_accuracies: list[list[float]], model_names: list[str], colors: dict[str, str]) -> None:
    plt.figure(figsize=(len(model_names), 3))
    # plot boxplot of per_seed_accuracy with per-group colors
    bp = plt.boxplot(
        per_seed_accuracies,
        tick_labels=model_names,
        patch_artist=True,  # allows filled boxes
        boxprops=dict(linewidth=1.2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        medianprops=dict(color="black", linewidth=1.4),
    )
    for patch, c in zip(bp["boxes"], colors.values()):
        patch.set_facecolor(c)
        patch.set_edgecolor("black")
    plt.xticks([])
    plt.ylabel("Accuracy")
    plt.title("Predicted miRNA\naverage per-seed accuracy")
    save_path = os.path.join(PROJ_HOME, "Performance", "TargetScan_test", "TwoTowerTransformer", "generated_mirna_average_per_seed_accuracy_mrna_length_comparison.svg")
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    print(f"Figure saved to {save_path}")

def main() -> None:
    DATASET_PATH1 = os.path.join(PROJ_HOME, "TargetScan_dataset", "generated_mirna_positive_samples_30_randomized_start_test.csv")
    DATASET_PATH2 = os.path.join(PROJ_HOME, "TargetScan_dataset", "generated_mirna_positive_primates_test_100_randomized_start_local_self_attn_full_cross_attn.csv")
    DATASET_PATH3 = os.path.join(PROJ_HOME, "TargetScan_dataset", "generated_mirna_positive_primates_test_500_randomized_start_local_self_attn_full_cross_attn.csv")
    
    dataframe1 = pd.read_csv(DATASET_PATH1)
    dataframe2 = pd.read_csv(DATASET_PATH2)
    dataframe3 = pd.read_csv(DATASET_PATH3)
    metrics1, per_seed_accuracy1 = compute_accuracy(dataframe1)
    metrics2, per_seed_accuracy2 = compute_accuracy(dataframe2)
    metrics3, per_seed_accuracy3 = compute_accuracy(dataframe3)

    plot_accuracy([metrics1, metrics2, metrics3], 
                ["30 nt", "100 nt", "500 nt"], 
                {"color1":"#FAB796", "color2": "#87CEBF", "color3": "#A1A9AD"})
    plot_per_seed_accuracy([per_seed_accuracy1, per_seed_accuracy2, per_seed_accuracy3], 
                ["30 nt", "100 nt", "500 nt"], 
                {"color1":"#FAB796", "color2": "#87CEBF", "color3": "#A1A9AD"})

    # # do mann-whitney u-test between per_seed_accuracy1 and per_seed_accuracy2
    # result = mannwhitneyu(np.log10(per_seed_accuracy1), np.log10(per_seed_accuracy2), alternative="two-sided", method="asymptotic")
    # print(f"Mann-Whitney u-test statistic: {result.statistic}")
    # print(f"Mann-Whitney u-test p-value: {mp.nstr(result.pvalue, 4)}")

if __name__ == "__main__":
    main()
