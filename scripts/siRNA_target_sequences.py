"""
Prepare FDA-approved siRNA data for zero-shot evaluation.

1. Define sense (mRNA target) and antisense (guide) strands from Traber & Yu 2024, Table 2
2. Fetch full mRNA sequences from NCBI
3. Locate the binding site on the mRNA using the sense strand
4. Extract 50nt window centered on the binding site
5. Save as CSV with columns: drug, gene_name, noncodingRNA, gene, Start, End

The 'noncodingRNA' column contains the antisense (guide) strand in 5'→3' orientation
(same convention as miRNA), converted from RNA (U) to DNA (T).
The 'gene' column contains the 50nt mRNA target region in DNA.
"""

import os
import pandas as pd
from Bio import Entrez, SeqIO

Entrez.email = "your email" 

# ══════════════════════════════════════════════════════════════════════════════
# 1.  FDA siRNA data from Traber & Yu 2024, Table 2
#     Antisense strands are written 3'→5' in the paper.
#     We reverse them to 5'→3' (standard miRNA convention).
# ══════════════════════════════════════════════════════════════════════════════

FDA_SIRNA = [
    {
        "drug": "Patisiran",
        "gene_name": "TTR",
        "accession": "NM_000371.4",
        "region": "3UTR",
        "region_start_nt": 46,
        "region_end_nt": 66,
        # Sense 5'→3' (= mRNA target sequence, as RNA)
        "sense_rna": "AUGUAACCAAGAGUAUUCCAU",
        # Antisense 3'→5' (as written in the paper)
        "antisense_3to5_rna": "UUCAUUGGUUCUCAUAAGGUA",
    },
    {
        "drug": "Givosiran",
        "gene_name": "ALAS1",
        "accession": "NM_000688.6",
        "region": "CDS",
        "region_start_nt": 534,
        "region_end_nt": 556,
        "sense_rna": "ACCAGAAAGAGUGUCUCAUCUUC",
        "antisense_3to5_rna": "UGGUCUUUCUCACAGAGUAGAAU",
    },
    {
        "drug": "Lumasiran",
        "gene_name": "HAO1",
        "accession": "NM_017545.3",
        "region": "3UTR",
        "region_start_nt": 204,
        "region_end_nt": 226,
        "sense_rna": "UGGACUUUCAUCCUGGAAAUAUA",
        "antisense_3to5_rna": "ACCUGAAAGUAGGACCUUUAUAU",
    },
    {
        "drug": "Inclisiran",
        "gene_name": "PCSK9",
        "accession": "NM_174936.4",
        "region": "3UTR",
        "region_start_nt": 1160,
        "region_end_nt": 1182,
        "sense_rna": "UUCUAGACCUGUUUUGCUUUUGU",
        "antisense_3to5_rna": "AAGAUCUGGACAAAACGAAAACA",
    },
    {
        "drug": "Vutrisiran",
        "gene_name": "TTR",
        "accession": "NM_000371.4",
        "region": "3UTR",
        "region_start_nt": 35,
        "region_end_nt": 57,
        "sense_rna": "GAUGGGAUUUCAUGUAACCAAGA",
        "antisense_3to5_rna": "CUACCCUAAAGUACAUUGGUUCU",
    },
    {
        "drug": "Nedosiran",
        "gene_name": "LDHA",
        "accession": "NM_005566.4",
        "region": "3UTR",
        "region_start_nt": 63,
        "region_end_nt": 84,
        "sense_rna": "GCAUGUUGUCCUUUUUAUCUGA",
        "antisense_3to5_rna": "GGUACAACAGGAAAAAGAGACU",
    },
]


def rna_to_dna(seq):
    """Convert RNA sequence (U) to DNA (T)."""
    return seq.replace("U", "T")


def reverse_string(seq):
    """Reverse a string."""
    return seq[::-1]


def reverse_complement_dna(seq):
    """Reverse complement of a DNA sequence."""
    comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    return "".join(comp[b] for b in reversed(seq))


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Fetch mRNA sequences from NCBI
# ══════════════════════════════════════════════════════════════════════════════

def fetch_mrna_from_ncbi(accession):
    """Fetch full mRNA sequence from NCBI Nucleotide database."""
    print(f"  Fetching {accession} from NCBI...", end=" ", flush=True)
    handle = Entrez.efetch(
        db="nucleotide",
        id=accession,
        rettype="fasta",
        retmode="text",
    )
    record = SeqIO.read(handle, "fasta")
    handle.close()
    seq = str(record.seq)
    print(f"{len(seq)} nt")
    return seq


def find_region_start(accession):
    """
    Fetch GenBank record to find the start position of CDS and 3'UTR
    on the full mRNA.
    """
    handle = Entrez.efetch(
        db="nucleotide",
        id=accession,
        rettype="gb",
        retmode="text",
    )
    record = SeqIO.read(handle, "genbank")
    handle.close()

    cds_start, cds_end = None, None
    for feature in record.features:
        if feature.type == "CDS":
            cds_start = int(feature.location.start)
            cds_end = int(feature.location.end)
            break

    if cds_start is None:
        print(f"  WARNING: No CDS found for {accession}")
        return {"CDS": 0, "3UTR": 0, "5UTR": 0}

    # 3'UTR starts right after CDS ends
    utr3_start = cds_end
    utr5_end = cds_start

    print(f"  {accession}: CDS={cds_start}-{cds_end}, 3'UTR starts at {utr3_start}")
    return {
        "CDS": cds_start,
        "3UTR": utr3_start,
        "5UTR": 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Main processing
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Cache fetched sequences to avoid redundant NCBI calls
    mrna_cache = {}      # accession → full DNA sequence
    region_cache = {}    # accession → {CDS: pos, 3UTR: pos}

    rows = []

    for entry in FDA_SIRNA:
        drug = entry["drug"]
        gene_name = entry["gene_name"]
        accession = entry["accession"]
        region = entry["region"]
        region_start = entry["region_start_nt"]
        region_end = entry["region_end_nt"]

        print(f"\n{'─'*60}")
        print(f"  {drug} ({gene_name}, {accession})")

        # ── Fetch mRNA ────────────────────────────────────────────
        if accession not in mrna_cache:
            mrna_cache[accession] = fetch_mrna_from_ncbi(accession)
            region_cache[accession] = find_region_start(accession)

        full_mrna_dna = mrna_cache[accession]
        region_offsets = region_cache[accession]

        # ── Convert sense strand to DNA and locate on mRNA ────────
        sense_dna = rna_to_dna(entry["sense_rna"])
        print(f"  Sense (DNA):  {sense_dna}")

        # Search for exact match in full mRNA
        pos = full_mrna_dna.find(sense_dna)

        if pos < 0:
            # Try with U→T already done, also try case-insensitive
            pos = full_mrna_dna.upper().find(sense_dna.upper())

        if pos < 0:
            # Calculate expected position from region annotation
            offset = region_offsets.get(region, 0)
            expected_pos = offset + region_start - 1  # 1-indexed to 0-indexed
            print(f"  WARNING: Exact sense match not found!")
            print(f"  Expected position (from annotation): {expected_pos}")
            print(f"  mRNA at expected pos: {full_mrna_dna[expected_pos:expected_pos+len(sense_dna)]}")
            pos = expected_pos
        else:
            print(f"  Found sense strand at position {pos}-{pos+len(sense_dna)} on full mRNA")

        # ── Verify against annotated position ─────────────────────
        offset = region_offsets.get(region, 0)
        expected_pos = offset + region_start - 1
        if pos != expected_pos:
            print(f"  NOTE: Found at {pos}, expected at {expected_pos} "
                  f"(offset={offset}, region_start={region_start})")

        # ── Extract 50nt window centered on binding site ──────────
        site_len = len(sense_dna)
        center = pos + site_len // 2
        window_start = max(0, center - 25)
        window_end = min(len(full_mrna_dna), center + 25)

        # Ensure exactly 50nt
        if window_end - window_start < 50 and window_start > 0:
            window_start = max(0, window_end - 50)
        if window_end - window_start < 50:
            window_end = min(len(full_mrna_dna), window_start + 50)

        target_50nt = full_mrna_dna[window_start:window_end]
        print(f"  50nt window [{window_start}:{window_end}]: {target_50nt}")
        print(f"  Window length: {len(target_50nt)} nt")

        # ── Convert antisense to 5'→3' DNA ────────────────────────
        # Paper gives antisense 3'→5', reverse to get 5'→3'
        antisense_5to3_rna = reverse_string(entry["antisense_3to5_rna"])
        antisense_5to3_dna = rna_to_dna(antisense_5to3_rna)
        print(f"  Antisense 5'→3' (DNA): {antisense_5to3_dna}")
        print(f"  Antisense length: {len(antisense_5to3_dna)} nt")

        # ── Verify complementarity ────────────────────────────────
        rc_of_antisense = reverse_complement_dna(antisense_5to3_dna)
        n_match = sum(1 for a, b in zip(rc_of_antisense, sense_dna) if a == b)
        print(f"  Complementarity check: {n_match}/{len(sense_dna)} matches "
              f"({n_match/len(sense_dna)*100:.1f}%)")

        # ── Build row ─────────────────────────────────────────────
        rows.append({
            "drug": drug,
            "gene_name": gene_name,
            "accession": accession,
            "region": region,
            "noncodingRNA": antisense_5to3_dna,    # antisense strand, 5'→3', DNA
            "gene": target_50nt,                   # 50nt mRNA target, DNA
            "Start": window_start,                 # 0-indexed on full mRNA
            "End": window_end,                     # 0-indexed on full mRNA
            "sense_strand": sense_dna,             # original sense strand
            "binding_start": pos,                  # exact binding start on full mRNA
            "binding_end": pos + site_len,
        })

    # ── Save ──────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    print(f"\n{'═'*60}")
    print(f"  Final DataFrame:")
    print(df[["drug", "gene_name", "noncodingRNA", "gene", "Start", "End"]].to_string())

    save_path = "fda_sirna_targets.csv"
    df.to_csv(save_path, index=False)
    print(f"\n  Saved to {save_path}")

    return df


if __name__ == "__main__":
    df = main()
