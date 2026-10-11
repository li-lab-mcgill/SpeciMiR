# MiRformer: a dual-transformer encoder framework for predicting miRNA-mRNA interactions

This repository contains the code for `MiRformer`, a deep learning model for predicting miRNA-mRNA interactions using dual-transformer encoder architecture.

## Datasets availability

All training, validation and testing data are available on [Zenodo](https://zenodo.org/records/23303444)

## Installation

MiRformer uses a conda environment named `SpeciMiR`. The environment is Python 3.10.16 with PyTorch 2.5.1 built for CUDA 12.4. Install [Miniconda or Anaconda](https://docs.conda.io/en/latest/miniconda.html) first, and use an NVIDIA driver that supports CUDA 12.4.

Create and activate the environment:

```bash
conda create -n SpeciMiR python=3.10.16 -y
conda activate SpeciMiR
```

Install PyTorch, torchvision, the CUDA 12.4 runtime, and Captum from the pytorch channel:

```bash
conda install -y -c pytorch -c nvidia \
    pytorch==2.5.1 \
    torchvision==0.20.1 \
    pytorch-cuda=12.4 \
    captum==0.8.0
```

Install ViennaRNA 2.4.7. The `py27_2` build depends only on `libgcc` and supplies the `RNAfold`, `RNAcofold`, and `RNAduplex` binaries used by the thermodynamic scoring scripts:

```bash
conda install -y -c bioconda -c conda-forge viennarna=2.4.7=py27_2
```

Install the remaining Python packages. These are the versions of the packages listed in `requirements.txt` and `torch-req.txt`, plus libraries imported by the training and plotting scripts (`wandb`, `seaborn`, `datasets`, and `logomaker`):

```bash
pip install \
    triton==3.1.0 \
    einops==0.8.0 \
    biopython==1.79 \
    "transformers==4.26.1" \
    omegaconf==2.3.0 \
    scikit-learn==1.6.1 \
    numpy==1.26.4 \
    pandas==2.2.3 \
    matplotlib==3.10.6 \
    deeplift==0.6.13.0 \
    shap==0.46.0 \
    scipy==1.15.1 \
    wandb==0.19.6 \
    seaborn==0.13.2 \
    datasets==3.5.1 \
    logomaker==0.8.6
```

`biopython` is pinned to 1.79 because several scripts import `Bio.pairwise2`.

Check that PyTorch sees CUDA 12.4 and that the ViennaRNA binaries are on `PATH`:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
which RNAfold RNAcofold RNAduplex
```

Activate `SpeciMiR` before the commands below.

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

