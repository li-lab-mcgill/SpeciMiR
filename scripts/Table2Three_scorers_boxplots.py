'''
This script plots boxplots of scores from three scorers: Discriminator, RNACofold, and RNAhybrid.
The scores are loaded from Manakov2022/Table2Three_scorers_scores_`dataset_name`_`score_type`.tsv.gz files.
Each boxplot plots the 4 score distributions for 4 datasets: real_miRNA_target_mRNA, generated_miRNA_target_mRNA, generated_miRNA_off_target_mRNA, and random_miRNA_target_mRNA.
'''

import os
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from scipy.stats import mannwhitneyu
from Global_parameters import PROJ_HOME, AXIS_FONT_SIZE, TICK_FONT_SIZE, TITLE_FONT_SIZE, LEGEND_FONT_SIZE
gill_sans_font = font_manager.FontProperties(family='Gill Sans')
plt.rcParams['font.family'] = gill_sans_font.get_name()

DATASET_NAMES = ['real_miRNA_target_mRNA', 'generated_miRNA_target_mRNA', 'generated_miRNA_off_target_mRNA', 'random_miRNA_target_mRNA']
SCORE_TYPES = ['Discriminator', 'RNACofold', 'RNAHybrid']
SCORE_LABELS = ['Binding probability', 'Delta G (kcal/mol)', 'Minimum free energy (kcal/mol)']
DATASET_COLORS = list(plt.get_cmap("Set2").colors[:len(DATASET_NAMES)])
PAIRWISE_COMPARISONS = list(combinations(range(len(DATASET_NAMES)), 2))
PVALUE_CORRECTION_METHOD = "bonferroni"


def pvalue_to_stars(p_value):
    if np.isnan(p_value):
        return "n/a"
    if p_value <= 0.001:
        return "***"
    if p_value <= 0.01:
        return "**"
    if p_value <= 0.05:
        return "*"
    return "ns"


def correct_pvalues(p_values):
    corrected = np.full(len(p_values), np.nan, dtype=float)
    finite_mask = np.isfinite(p_values)
    if not np.any(finite_mask):
        return corrected

    finite_values = np.asarray(p_values)[finite_mask]
    if PVALUE_CORRECTION_METHOD == "bonferroni":
        corrected_values = np.minimum(finite_values * len(finite_values), 1.0)
    else:
        raise ValueError(f"Unsupported p-value correction method: {PVALUE_CORRECTION_METHOD}")

    corrected[finite_mask] = corrected_values
    return corrected


def annotate_pairwise_significance(ax, score_type, score_distributions):
    finite_distributions = [scores[np.isfinite(scores)] for scores in score_distributions]
    finite_distributions = [scores for scores in finite_distributions if len(scores) > 0]
    if not finite_distributions:
        return

    data_min = min(np.min(scores) for scores in finite_distributions)
    data_max = max(np.max(scores) for scores in finite_distributions)
    data_range = data_max - data_min
    if data_range == 0:
        data_range = max(abs(data_max), 1.0)

    bracket_height = data_range * 0.03
    bracket_step = data_range * 0.12
    base_y = data_max + data_range * 0.08

    raw_pvalues = []
    valid_pairs = []
    for left_idx, right_idx in PAIRWISE_COMPARISONS:
        left_scores = score_distributions[left_idx]
        right_scores = score_distributions[right_idx]
        left_scores = left_scores[np.isfinite(left_scores)]
        right_scores = right_scores[np.isfinite(right_scores)]

        if len(left_scores) == 0 or len(right_scores) == 0:
            p_value = np.nan
        else:
            p_value = mannwhitneyu(left_scores, right_scores, alternative="two-sided", method="asymptotic").pvalue
        raw_pvalues.append(p_value)
        valid_pairs.append((left_idx, right_idx))

    corrected_pvalues = correct_pvalues(raw_pvalues)

    for level, ((left_idx, right_idx), raw_pvalue, corrected_pvalue) in enumerate(zip(valid_pairs, raw_pvalues, corrected_pvalues)):
        star_label = pvalue_to_stars(corrected_pvalue)
        print(
            f"{score_type}: {DATASET_NAMES[left_idx]} vs {DATASET_NAMES[right_idx]} "
            f"raw p-value = {raw_pvalue:.4e}, corrected p-value = {corrected_pvalue:.4e} "
            f"({PVALUE_CORRECTION_METHOD}, {star_label})"
        )

        y = base_y + level * bracket_step
        ax.plot(
            [left_idx, left_idx, right_idx, right_idx],
            [y, y + bracket_height, y + bracket_height, y],
            color="black",
            linewidth=1.2,
            clip_on=False,
        )
        ax.text(
            (left_idx + right_idx) / 2,
            y + bracket_height,
            star_label,
            ha="center",
            va="bottom",
            fontsize=TICK_FONT_SIZE,
            clip_on=False,
        )

    ax.set_ylim(data_min - data_range * 0.05, base_y + len(PAIRWISE_COMPARISONS) * bracket_step + bracket_height + data_range * 0.08)

fig, subplots = plt.subplots(1, len(SCORE_TYPES), figsize=(18, 7))
for i, (ax, score_type) in enumerate(zip(subplots, SCORE_TYPES)):
    score_distributions = []
    for j, dataset_name in enumerate(DATASET_NAMES):
        score_file = os.path.join(PROJ_HOME, "Manakov2022", "Table2Three_scorers_scores_{}_{}.tsv.gz".format(dataset_name, score_type))
        df = pd.read_csv(score_file, compression="gzip", sep="\t")
        scores = df["score"].to_numpy()
        score_distributions.append(scores)
        boxplot = ax.boxplot(scores, positions=[j], widths=0.7, patch_artist=True)
        color = DATASET_COLORS[j]
        boxplot["boxes"][0].set_facecolor(color)
        boxplot["boxes"][0].set_edgecolor(color)
        for key in ["whiskers", "caps", "medians"]:
            for artist in boxplot[key]:
                artist.set_color(color)
    ax.set_title(score_type, fontsize=TITLE_FONT_SIZE)
    ax.set_xlabel("Dataset", fontsize=AXIS_FONT_SIZE)
    ax.set_xticks(range(len(DATASET_NAMES)))
    ax.set_xticklabels(DATASET_NAMES, rotation=30, ha="right", fontsize=TICK_FONT_SIZE)
    ax.set_ylabel(f"{SCORE_LABELS[i]}", fontsize=AXIS_FONT_SIZE)
    annotate_pairwise_significance(ax, score_type, score_distributions)

fig.tight_layout()
fig.savefig(os.path.join(PROJ_HOME, "Manakov2022", "Table2Three_scorers_boxplots.svg"), dpi=300)
print("Table2Three_scorers_boxplots.svg saved to: ", os.path.join(PROJ_HOME, "Manakov2022", "Table2Three_scorers_boxplots.svg"))
