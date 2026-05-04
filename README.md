# SpeciMiR---Target-Conditioned-MicroRNA-Synthesis-via-Specificity-Guided-Transformers

This repository contains the code for `SpeciMiR`, a deep learning model for predicting miRNA-mRNA interactions using dual-transformer encoder architecture.

## Command-line Usage

The main entry point for training and evaluation is `scripts/DTEA_model.py`.

### Evaluation

To test MiRformer model, use the following command. This example uses the `DTEA` base model with a custom configuration and a minimal test dataset.

```bash
python scripts/eval_DTEA_model.py \
    --mirna_max_len 24 \
    --mrna_max_len 520 \
    --device "cuda" \
    --embed_dim 1024 \
    --ff_dim 4096 \
    --num_heads 8 \
    --num_layers 4 \
    --predict_span \
    --predict_binding \
    --predict_cleavage \
    --use_longformer \
    --test_path "test_data/sample_data.csv"
```

**Arguments:**

- `--mirna_max_len`: max mirna length.
- `--mrna_max_len`: max mrna length.
- `--device`: Device to use (e.g., `cpu`, `cuda`).
- `--embed_dim`: embedding dimension, default to 1024.
- `--ff_dim`: feed-forward dimension, default to 4096.
- `--num_heads`: number of head to use in each transformer encoder.
- `--num_layers`: number of layers to use in each transformer encoder.
- `--predict_span`: flag to be set True for model to predict seed region spans in mRNA inputs.
- `--predict_binding`: flag to be set True for model to predict interaction between miRNA-mRNA pairs.
- `--predict_cleavage`: flag to be set True for model to predict cleavage sites in mRNA inputs.
- `--use_longformer`: flag to be set True when using longformers. When false, standard multi-head transformers are used. 
- `test_path`: path to test dataset


## Minimal Test Dataset

A minimal test dataset is provided in `test_data/sample_data.csv` to validate that the software runs correctly.

**Location:** `test_data/sample_data.csv`

**Format:**
The CSV file requires the following columns:
- `Transcript ID`
- `miRNA ID`
- `miRNA sequence`
- `mRNA sequence`
- `seed start`
- `seed end`
- `label` (0 for negative, 1 for positive)

