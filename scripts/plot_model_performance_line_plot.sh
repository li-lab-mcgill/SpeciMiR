#!/bin/bash
python scripts/plot_model_performance_line_plot.py \
    --mRNA_max_len 500\
    --dataset_name TargetScan\
    --model_dirs HyenaDNA_miRNA_500 \
    --model_names Finetuned-HyenaDNA-500nt \
    --train_loss_save_path /path/to/train_loss.png \
    --test_acc_save_path /path/to/evaluation_acc.png \
    --test_loss_save_path /path/to/evaluation_loss.png \
