import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from Global_parameters import PROJ_HOME, TICK_FONT_SIZE, AXIS_FONT_SIZE, TITLE_FONT_SIZE, LEGEND_FONT_SIZE

def get_complement(seq):
    """Returns the Watson-Crick complement of a sequence (DNA/RNA)."""
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return "".join(complement.get(base, base) for base in seq)

def verify_seed_matches(mrna_seq, mirna_seq):
    """
    Scans mRNA for 8-mer, 7-mer, and 6-mer matches to the miRNA seed.
    miRNA seed is defined as positions 2-8.
    """
    results = {
        'match_found': False,
        'match_type': None,
        'start_pos': -1
    }

    # reverse mirna sequence from (3' to 5') to (5' to 3')
    mirna_seq = mirna_seq[::-1]
    seed_segment = mirna_seq[1:8]
    seed_rc = get_complement(seed_segment)[::-1] # reverse complement of seed segment because mirna and mrna are anti-parallel
    # miRNA is 1-indexed in literature: pos 2-8
    # In Python 0-indexing: pos 1 to 8 (exclusive)
    seed_8mer = seed_rc # 7 bases
    seed_7mer = seed_rc[1:] # 6 bases
    seed_6mer = seed_rc[1:] # 6 bases
    
    # 1. Check for 8-mer (Match to seed 2-8 AND an 'A' at mRNA position 1)
    # This requires looking for: [Reverse Complement of 2-8] + [A]
    target_8mer = seed_8mer + "A"
    if target_8mer in mrna_seq:
        return True, "8-mer", mrna_seq.find(target_8mer)

    # 2. Check for 7-mer-m8 (Exact match to seed 2-8)
    if seed_8mer in mrna_seq:
        return True, "7-mer-m8", mrna_seq.find(seed_8mer)

    # 3. Check for 7-mer-A1 (Match to seed 2-7 AND an 'A' at mRNA position 1)
    target_7mer_a1 = seed_7mer + "A"
    if target_7mer_a1 in mrna_seq:
        return True, "7-mer-A1", mrna_seq.find(target_7mer_a1)

    # 4. Check for 6-mer (Match to seed 2-7)
    if seed_6mer in mrna_seq:
        return True, "6-mer", mrna_seq.find(seed_6mer)

    return False, None, -1

def verify_anywhere_seed_matches(mrna_seq, mirna_seq):
    """
    Slides a window across the entire miRNA (5'→3') and checks whether the
    reverse complement of each window appears in the mRNA, looking for the
    strongest match type anywhere along the miRNA (not only at seed positions 2-8).

    Match type priority (strongest to weakest):
        8-mer    : RC of any 7-nt miRNA window + 'A' anchor found in mRNA
        7-mer-m8 : RC of any 7-nt miRNA window found in mRNA
        7-mer-A1 : RC of any 6-nt miRNA window + 'A' anchor found in mRNA
        6-mer    : RC of any 6-nt miRNA window found in mRNA

    Returns:
        (found: bool,
         match_type: str or None,
         mirna_start: int,   # 0-based window start in the 5'→3' miRNA
         mrna_pos: int)      # 0-based position in mRNA where match begins
    """
    # Convert stored 3'→5' representation to 5'→3'
    mirna_fwd = mirna_seq[::-1]
    mirna_len = len(mirna_fwd)

    TYPE_PRIORITY = {"8-mer": 4, "7-mer-m8": 3, "7-mer-A1": 2, "6-mer": 1}
    best_type      = None
    best_priority  = 0
    best_mirna_pos = -1
    best_mrna_pos  = -1

    for i in range(mirna_len):
        # ── 7-nt window → 8-mer and 7-mer-m8 ──────────────────────────────
        if i + 7 <= mirna_len:
            rc_7 = get_complement(mirna_fwd[i:i+7])[::-1]

            if TYPE_PRIORITY["8-mer"] > best_priority:
                target = rc_7 + "A"
                if target in mrna_seq:
                    best_type, best_priority = "8-mer", TYPE_PRIORITY["8-mer"]
                    best_mirna_pos, best_mrna_pos = i, mrna_seq.find(target)

            if TYPE_PRIORITY["7-mer-m8"] > best_priority and rc_7 in mrna_seq:
                best_type, best_priority = "7-mer-m8", TYPE_PRIORITY["7-mer-m8"]
                best_mirna_pos, best_mrna_pos = i, mrna_seq.find(rc_7)

        # ── 6-nt window → 7-mer-A1 and 6-mer ──────────────────────────────
        if i + 6 <= mirna_len:
            rc_6 = get_complement(mirna_fwd[i:i+6])[::-1]

            if TYPE_PRIORITY["7-mer-A1"] > best_priority:
                target = rc_6 + "A"
                if target in mrna_seq:
                    best_type, best_priority = "7-mer-A1", TYPE_PRIORITY["7-mer-A1"]
                    best_mirna_pos, best_mrna_pos = i, mrna_seq.find(target)

            if TYPE_PRIORITY["6-mer"] > best_priority and rc_6 in mrna_seq:
                best_type, best_priority = "6-mer", TYPE_PRIORITY["6-mer"]
                best_mirna_pos, best_mrna_pos = i, mrna_seq.find(rc_6)

        # Early exit once the strongest possible match is found
        if best_priority == TYPE_PRIORITY["8-mer"]:
            break

    if best_type is not None:
        return True, best_type, best_mirna_pos, best_mrna_pos
    return False, None, -1, -1


def plot_seed_match_distribution(data, plot_path):
    # Normalized Data for 100% stacked bar chart
    # This helps in comparing the distribution of match types across different mRNA lengths
    data_perc = data.div(data.sum(axis=0), axis=1) * 100
    lengths = data.columns.tolist()
    six_mers = data.loc["6-mer"].tolist()
    seven_a1 = data.loc["7-mer-A1"].tolist()
    seven_m8 = data.loc["7-mer-m8"].tolist()
    eight_mers = data.loc["8-mer"].tolist()

    six_mers_perc = data_perc.loc["6-mer"].tolist()
    seven_a1_perc = data_perc.loc["7-mer-A1"].tolist()
    seven_m8_perc = data_perc.loc["7-mer-m8"].tolist()
    eight_mers_perc = data_perc.loc["8-mer"].tolist()

    # Plotting Raw Counts
    tab20b = plt.get_cmap("tab20b").colors
    bar_colors = [tab20b[0], tab20b[1], tab20b[2], tab20b[3]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # Subplot 1: Raw Counts
    ax1.bar(lengths, six_mers, label='6-mer', color=bar_colors[0])
    ax1.bar(lengths, seven_a1, bottom=six_mers, label='7-mer-A1', color=bar_colors[1])
    ax1.bar(lengths, seven_m8, bottom=np.asarray(six_mers)+np.asarray(seven_a1), label='7-mer-m8', color=bar_colors[2])
    ax1.bar(lengths, eight_mers, bottom=np.asarray(six_mers)+np.asarray(seven_a1)+np.asarray(seven_m8), label='8-mer', color=bar_colors[3])

    ax1.set_title("Seed Match Counts", fontsize=TITLE_FONT_SIZE)
    ax1.legend(fontsize=LEGEND_FONT_SIZE)
    ax1.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    ax1.set_xlabel("mRNA length", fontsize=AXIS_FONT_SIZE)
    ax1.set_ylabel("Number of Matches", fontsize=AXIS_FONT_SIZE)

    # Subplot 2: Percentages (100% Stacked)
    ax2.bar(lengths, six_mers_perc, label='6-mer', color=bar_colors[0])
    ax2.bar(lengths, seven_a1_perc, bottom=six_mers_perc, label='7-mer-A1', color=bar_colors[1])
    ax2.bar(lengths, seven_m8_perc, bottom=np.asarray(six_mers_perc)+np.asarray(seven_a1_perc), label='7-mer-m8', color=bar_colors[2])
    ax2.bar(lengths, eight_mers_perc, bottom=np.asarray(six_mers_perc)+np.asarray(seven_a1_perc)+np.asarray(seven_m8_perc), label='8-mer', color=bar_colors[3])

    ax2.set_ylabel("Percentage (%)", fontsize=AXIS_FONT_SIZE)
    ax2.set_title("Seed Match Distribution (%)", fontsize=TITLE_FONT_SIZE)
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=LEGEND_FONT_SIZE)
    ax2.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    ax2.set_xlabel("mRNA length", fontsize=AXIS_FONT_SIZE)

    # save the plots
    fig.savefig(plot_path, dpi=400, bbox_inches="tight")
    print("Seed match distribution plot saved to: ", plot_path)

if __name__ == "__main__":
    # read in the dataset
    data_dir = os.path.join(PROJ_HOME, "TargetScan_dataset")
    data_30nt = os.path.join(data_dir, "generated_mirna_seed_perturbation_30_random_samples_test.csv")
    data_100nt = os.path.join(data_dir, "generated_mirna_seed_perturbation_120_longformer_random_samples_test.csv")
    data_500nt = os.path.join(data_dir, "generated_mirna_seed_perturbation_520_longformer_random_samples_test.csv")
    data_30nt = pd.read_csv(data_30nt)
    data_100nt = pd.read_csv(data_100nt)
    data_500nt = pd.read_csv(data_500nt)
    valid_6mers_30nt = 0
    valid_7mers_a1_30nt = 0
    valid_7mers_m8_30nt = 0
    valid_8mers_30nt = 0
    valid_6mers_100nt = 0
    valid_7mers_a1_100nt = 0
    valid_7mers_m8_100nt = 0
    valid_8mers_100nt = 0
    valid_6mers_500nt = 0
    valid_7mers_a1_500nt = 0
    valid_7mers_m8_500nt = 0
    valid_8mers_500nt = 0

    # 30nt
    # verify the seed matches
    for index, row in data_30nt.iterrows():
        mrna_seq = row["mRNA sequence"]
        mirna_seq = row["generated_mirna"]
        found, m_type, _, _ = verify_anywhere_seed_matches(mrna_seq, mirna_seq) # can switch to verify_seed_matches for exact seed matches at 2-8nt
        if found:
            if m_type == "6-mer":
                valid_6mers_30nt += 1
            elif m_type == "7-mer-A1":
                valid_7mers_a1_30nt += 1
            elif m_type == "7-mer-m8":
                valid_7mers_m8_30nt += 1
            elif m_type == "8-mer":
                valid_8mers_30nt += 1
    print(f"Valid 6-mers: {valid_6mers_30nt}")
    print(f"Valid 7-mers-A1: {valid_7mers_a1_30nt}")
    print(f"Valid 7-mers-m8: {valid_7mers_m8_30nt}")
    print(f"Valid 8-mers: {valid_8mers_30nt}")

    # percentage of valid 6-mers, 7-mers-A1, 7-mers-m8, and 8-mers
    total_matches = valid_6mers_30nt + valid_7mers_a1_30nt + valid_7mers_m8_30nt + valid_8mers_30nt
    print(f"Total matches: {total_matches / len(data_30nt):.2%}") # percentage of total matches
    print(f"Percentage of valid 6-mers: {valid_6mers_30nt / total_matches:.2%}") # percentage of valid 6-mers
    print(f"Percentage of valid 7-mers-A1: {valid_7mers_a1_30nt / total_matches:.2%}") # percentage of valid 7-mers-A1
    print(f"Percentage of valid 7-mers-m8: {valid_7mers_m8_30nt / total_matches:.2%}") # percentage of valid 7-mers-m8
    print(f"Percentage of valid 8-mers: {valid_8mers_30nt / total_matches:.2%}") # percentage of valid 8-mers

    print("____________________________________________________________")

    # 100nt
    # verify the seed matches
    for index, row in data_100nt.iterrows():
        mrna_seq = row["mRNA sequence"]
        mirna_seq = row["generated_mirna"]
        found, m_type, _,_ = verify_anywhere_seed_matches(mrna_seq, mirna_seq) # can switch to verify_seed_matches for exact seed matches at 2-8nt
        if found:
            if m_type == "6-mer":
                valid_6mers_100nt += 1
            elif m_type == "7-mer-A1":
                valid_7mers_a1_100nt += 1
            elif m_type == "7-mer-m8":
                valid_7mers_m8_100nt += 1
            elif m_type == "8-mer":
                valid_8mers_100nt += 1
    print(f"Valid 6-mers: {valid_6mers_100nt}")
    print(f"Valid 7-mers-A1: {valid_7mers_a1_100nt}")
    print(f"Valid 7-mers-m8: {valid_7mers_m8_100nt}")
    print(f"Valid 8-mers: {valid_8mers_100nt}")

    # percentage of valid 6-mers, 7-mers-A1, 7-mers-m8, and 8-mers
    total_matches = valid_6mers_100nt + valid_7mers_a1_100nt + valid_7mers_m8_100nt + valid_8mers_100nt
    print(f"Total matches: {total_matches / len(data_100nt):.2%}") # percentage of total matches
    print(f"Percentage of valid 6-mers: {valid_6mers_100nt / total_matches:.2%}") # percentage of valid 6-mers
    print(f"Percentage of valid 7-mers-A1: {valid_7mers_a1_100nt / total_matches:.2%}") # percentage of valid 7-mers-A1
    print(f"Percentage of valid 7-mers-m8: {valid_7mers_m8_100nt / total_matches:.2%}") # percentage of valid 7-mers-m8
    print(f"Percentage of valid 8-mers: {valid_8mers_100nt / total_matches:.2%}") # percentage of valid 8-mers

    print("____________________________________________________________")

    # 500nt
    # verify the seed matches
    for index, row in data_500nt.iterrows():
        mrna_seq = row["mRNA sequence"]
        mirna_seq = row["generated_mirna"]
        found, m_type, _, _ = verify_anywhere_seed_matches(mrna_seq, mirna_seq) # can switch to verify_seed_matches for exact seed matches at 2-8nt
        if found:
            if m_type == "6-mer":
                valid_6mers_500nt += 1
            elif m_type == "7-mer-A1":
                valid_7mers_a1_500nt += 1
            elif m_type == "7-mer-m8":
                valid_7mers_m8_500nt += 1
            elif m_type == "8-mer":
                valid_8mers_500nt += 1
    print(f"Valid 6-mers: {valid_6mers_500nt}")
    print(f"Valid 7-mers-A1: {valid_7mers_a1_500nt}")
    print(f"Valid 7-mers-m8: {valid_7mers_m8_500nt}")
    print(f"Valid 8-mers: {valid_8mers_500nt}")

    # percentage of valid 6-mers, 7-mers-A1, 7-mers-m8, and 8-mers
    total_matches = valid_6mers_500nt + valid_7mers_a1_500nt + valid_7mers_m8_500nt + valid_8mers_500nt
    print(f"Total matches: {total_matches / len(data_500nt):.2%}") # percentage of total matches
    print(f"Percentage of valid 6-mers: {valid_6mers_500nt / total_matches:.2%}") # percentage of valid 6-mers
    print(f"Percentage of valid 7-mers-A1: {valid_7mers_a1_500nt / total_matches:.2%}") # percentage of valid 7-mers-A1
    print(f"Percentage of valid 7-mers-m8: {valid_7mers_m8_500nt / total_matches:.2%}") # percentage of valid 7-mers-m8
    print(f"Percentage of valid 8-mers: {valid_8mers_500nt / total_matches:.2%}") # percentage of valid 8-mers

    # plot seed match distribution
    lengths = ['30nt', '100nt', '500nt']
    seed_types = ["6-mer", "7-mer-A1", "7-mer-m8", "8-mer"]
    six_mers = [valid_6mers_30nt, valid_6mers_100nt, valid_6mers_500nt]
    seven_a1 = [valid_7mers_a1_30nt, valid_7mers_a1_100nt, valid_7mers_a1_500nt]
    seven_m8 = [valid_7mers_m8_30nt, valid_7mers_m8_100nt, valid_7mers_m8_500nt]
    eight_mers = [valid_8mers_30nt, valid_8mers_100nt, valid_8mers_500nt]

    # Normalized Data for 100% stacked bar chart
    # This helps in comparing the distribution of match types across different mRNA lengths
    data = pd.DataFrame(np.array([six_mers, seven_a1, seven_m8, eight_mers]), columns=lengths, index=seed_types)
    plot_path = os.path.join(PROJ_HOME, "Performance", "TargetScan_test", "TwoTowerTransformer", "seed_match_distribution_anywhere_after_perturbation.svg")
    plot_seed_match_distribution(data, plot_path)
