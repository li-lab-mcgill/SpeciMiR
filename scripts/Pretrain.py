import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp

import os
import math
import wandb
import random
import numpy as np
from time import time

from transformer_model import QuestionAnsweringModel
from multilevel_masking_pretraining import PairedPretrainWrapper, loss_stage1_baseline, loss_stage2_seed_mrna, loss_stage2_seed_mirna, loss_stage3_bispan
from Attention_regularization import kl_diag_seed_loss
from Data_pipeline import CharacterTokenizer, QuestionAnswerDataset
from utils import load_dataset
from ckpt_util import save_training_state, load_training_state

import contextlib
from wandb.sdk.wandb_settings import Settings

from ckpt_utils import (
    latest_checkpoint, save_training_state, load_training_state, GracefulKiller, is_rank0
)
from Global_parameters import PROJ_HOME

def setup(rank, world_size, seed=42):
    """Initialize the process group for distributed training."""
    # Initialize the process group
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)

    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    
    # Set synchronized random seed for reproducibility across all processes
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cleanup():
    """Clean up the process group."""
    dist.destroy_process_group()

def cosine_decay(total, step, min_factor=0.1):
    """
    Cosine decay from 1.0 at step 0 to min_factor at step total_updates-1.
    """
    if total < 1:
        return 1.0
    else:
        # cosine to min_factor
        progress = step / (total - 1)
        f = min_factor + 0.5 * (1 - min_factor) * (1 + math.cos(math.pi * progress))
        return f

def pretrain_loop(
            model: torch.nn.Module, 
            pretrain_wrapper: torch.nn.Module,
            dataloader: DataLoader, 
            optimizer: torch.optim, 
            scheduler: torch.optim.lr_scheduler, 
            device: str, 
            epoch: int, 
            total_updates: int, 
            updates_per_epoch: int,
            mask_id: int, 
            pad_id: int, 
            vocab_size: int,
            accumulation_step=1, 
            sigma=1.0,
            global_step=0,
            rank=0):
    model.train()
    total_loss = 0.0
    total_loss1 = 0.0
    total_loss2 = 0.0
    total_loss2_2 = 0.0
    total_loss3 = 0.0
    total_loss_reg = 0.0
    loss_list = []
    loss1_list = []
    loss2_list = []
    loss2_2_list = []
    loss3_list = []
    loss_reg_list = []
    pretrain_wrapper.train()
    for batch_idx, batch in enumerate(dataloader):
        for k in batch: batch[k] = batch[k].to(device)

        # === MLM forward ===
        logits_mr, logits_mi, attn_weights = pretrain_wrapper(
            mrna_ids=batch["mrna_input_ids"],
            mrna_mask=batch["mrna_attention_mask"],
            mirna_ids=batch["mirna_input_ids"],
            mirna_mask=batch["mirna_attention_mask"],
        )

        real_wrapper = getattr(pretrain_wrapper, "module", pretrain_wrapper)
        # === MLM loss (both) ===
        # Use the underlying module for loss calculations to avoid DDP wrapper issues
        loss1 = loss_stage1_baseline(wrapper=real_wrapper, batch=batch, pad_id=pad_id, mask_id=mask_id, vocab_size=vocab_size)  
        loss2 = loss_stage2_seed_mrna(wrapper=real_wrapper, batch=batch, pad_id=pad_id, mask_id=mask_id, vocab_size=vocab_size)
        loss2_2 = loss_stage2_seed_mirna(wrapper=real_wrapper, batch=batch, pad_id=pad_id, mask_id=mask_id, vocab_size=vocab_size)
        loss3 = loss_stage3_bispan(wrapper=real_wrapper, batch=batch, pad_id=pad_id, mask_id=mask_id, vocab_size=vocab_size)
        loss_mlm = loss1 + loss2 + loss2_2 + loss3

        # compute global update index (constant over an accumulation group)
        update_in_epoch = batch_idx // accumulation_step
        global_update = epoch * updates_per_epoch + update_in_epoch
        lamb = cosine_decay(total_updates, global_update, min_factor=0.1)
        loss = loss_mlm 

        # --- DDP-friendly backward with no_sync() for microbatches ---
        ddp_wrapper = pretrain_wrapper
        sync_now = ((batch_idx + 1) % accumulation_step == 0)
        ctx = ddp_wrapper.no_sync() if (hasattr(ddp_wrapper, "no_sync") and not sync_now) else contextlib.nullcontext()
        with ctx:
            (loss / accumulation_step).backward()
        bs = batch['mrna_input_ids'].size(0)
        
        # Accumulate individual losses for logging
        total_loss += loss.item() * accumulation_step
        total_loss += loss.item() * accumulation_step
        total_loss1 += loss1.item()
        total_loss2 += loss2.item()
        total_loss2_2 += loss2_2.item()
        total_loss3 += loss3.item()
        
        if accumulation_step != 1:
            loss_list.append((loss.item() / accumulation_step))
            loss1_list.append((loss1.item() / accumulation_step))
            loss2_list.append((loss2.item() / accumulation_step))
            loss2_2_list.append((loss2_2.item() / accumulation_step))
            loss3_list.append((loss3.item() / accumulation_step))
            if (batch_idx + 1) % accumulation_step == 0:
                torch.nn.utils.clip_grad_norm_(ddp_wrapper.parameters(), 5.0)
                optimizer.step()
                if scheduler is not None: scheduler.step()
                global_step += 1
                optimizer.zero_grad()
                
                print(
                    f"Train Epoch: {epoch} "
                    f"[{(batch_idx + 1) * bs}/{len(dataloader.dataset)} "
                    f"({(batch_idx + 1) * bs / len(dataloader.dataset) * 100:.0f}%)] "
                    f"loss1={sum(loss1_list):.3f} loss2={sum(loss2_list):.3f} loss2_2={sum(loss2_2_list):.3f} loss3={sum(loss3_list):.3f} "
                    f"Avg loss: {sum(loss_list):.6f}\n",
                    flush=True
                )
                loss_list = []
                loss1_list = []
                loss2_list = []
                loss2_2_list = []
                loss3_list = []
    
    # After the loop, if gradients remain (for non-divisible number of batches)
    if (batch_idx + 1) % accumulation_step != 0:
        torch.nn.utils.clip_grad_norm_(pretrain_wrapper.parameters(), 5.0)
        optimizer.step()
        if scheduler is not None: scheduler.step()
        global_step += 1
        optimizer.zero_grad()
    
    # Calculate epoch averages
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_loss1 = total_loss1 / num_batches
    avg_loss2 = total_loss2 / num_batches
    avg_loss2_2 = total_loss2_2 / num_batches
    avg_loss3 = total_loss3 / num_batches
    
    return {"Avg Loss": avg_loss, 
            "Loss1": avg_loss1, 
            "Loss2": avg_loss2, 
            "Loss2_2": avg_loss2_2,
            "Loss3": avg_loss3, 
            "global_step": global_step,
            }

@torch.no_grad()
def evaluate_mlm(model, 
                pretrain_wrapper,
                dataloader, 
                device, 
                vocab_size,
                pad_id,
                mask_id,
                rank=0):
    model.eval()
    pretrain_wrapper.eval()
    
    # Separate tracking for each stage
    total_loss1 = 0
    total_loss2 = 0
    total_loss2_2 = 0
    total_loss3 = 0
    total_correct1 = 0
    total_correct2 = 0
    total_correct2_2 = 0
    total_correct3 = 0
    total_masked1 = 0
    total_masked2 = 0
    total_masked2_2 = 0
    total_masked3 = 0
    
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}         
        
        real_wrapper = getattr(pretrain_wrapper, "module", pretrain_wrapper)
        loss1, correct1, mr_y1, mi_y1 = loss_stage1_baseline(wrapper=real_wrapper, batch=batch, pad_id=pad_id, mask_id=mask_id, vocab_size=vocab_size, evaluate=True)
        loss2, correct2, mr_y2 = loss_stage2_seed_mrna(wrapper=real_wrapper, batch=batch, pad_id=pad_id, mask_id=mask_id, vocab_size=vocab_size, evaluate=True)
        loss2_2, correct2_2, mi_y2_2 = loss_stage2_seed_mirna(wrapper=real_wrapper, batch=batch, pad_id=pad_id, mask_id=mask_id, vocab_size=vocab_size, evaluate=True)
        loss3, correct3, mr_y3, mi_y3 = loss_stage3_bispan(wrapper=real_wrapper, batch=batch, pad_id=pad_id, mask_id=mask_id, vocab_size=vocab_size, evaluate=True)
        
        # Accumulate losses
        total_loss1 += loss1.item()
        total_loss2 += loss2.item()
        total_loss2_2 += loss2_2.item()
        total_loss3 += loss3.item()
        
        # Accumulate correct predictions
        total_correct1 += correct1
        total_correct2 += correct2
        total_correct2_2 += correct2_2
        total_correct3 += correct3
        
        # Count masked tokens for each stage
        total_masked1 += mr_y1.ne(-100).sum().item() + mi_y1.ne(-100).sum().item()
        total_masked2 += mr_y2.ne(-100).sum().item()
        total_masked2_2 += mi_y2_2.ne(-100).sum().item()
        total_masked3 += mr_y3.ne(-100).sum().item() + mi_y3.ne(-100).sum().item()

    # Calculate averages
    avg_loss1 = total_loss1 / len(dataloader)
    avg_loss2 = total_loss2 / len(dataloader)
    avg_loss2_2 = total_loss2_2 / len(dataloader)
    avg_loss3 = total_loss3 / len(dataloader)
    
    # Calculate accuracies
    acc1 = total_correct1 / max(total_masked1, 1)  # Avoid division by zero
    acc2 = total_correct2 / max(total_masked2, 1)
    acc2_2 = total_correct2_2 / max(total_masked2_2, 1)
    acc3 = total_correct3 / max(total_masked3, 1)
    
    # Overall metrics
    total_loss = avg_loss1 + avg_loss2 + avg_loss2_2 + avg_loss3
    total_correct = total_correct1 + total_correct2 + total_correct2_2 + total_correct3
    total_masked = total_masked1 + total_masked2 + total_masked2_2 + total_masked3
    overall_acc = total_correct / max(total_masked, 1)
    
    if rank == 0:
        print(f"Evaluation Results:")
        print(f"  Stage 1 (Baseline): Accuracy={acc1:.4f}")
        print(f"  Stage 2 (Seed mRNA): Accuracy={acc2:.4f}")
        print(f"  Stage 2 (Seed miRNA): Accuracy={acc2_2:.4f}")
        print(f"  Stage 3 (Bispan): Accuracy={acc3:.4f}")
        print(f"  Overall: Loss={total_loss:.4f}, Accuracy={overall_acc:.4f}")
    
    return {
        'overall_loss': total_loss,
        'overall_acc': overall_acc,
        'stage1_acc': acc1,
        'stage2_acc': acc2,
        'stage2_2_acc': acc2_2,
        'stage3_acc': acc3
    }

@torch.no_grad()
def evaluate_mlm_ddp(model, 
                pretrain_wrapper,
                dataloader, 
                device, 
                vocab_size,
                pad_id,
                mask_id,
                rank=0):
    model.eval()
    pretrain_wrapper.eval()
    
    # Separate tracking for each stage - use tensors for all_reduce
    local_loss1 = torch.tensor(0.0, device=device)
    local_loss2 = torch.tensor(0.0, device=device)
    local_loss2_2 = torch.tensor(0.0, device=device)
    local_loss3 = torch.tensor(0.0, device=device)
    local_correct1 = torch.tensor(0, dtype=torch.long, device=device)
    local_correct2 = torch.tensor(0, device=device)
    local_correct2_2 = torch.tensor(0, dtype=torch.long, device=device)
    local_correct3 = torch.tensor(0, dtype=torch.long, device=device)
    local_masked1 = torch.tensor(0, dtype=torch.long, device=device)
    local_masked2 = torch.tensor(0, dtype=torch.long, device=device)
    local_masked2_2 = torch.tensor(0, dtype=torch.long, device=device)
    local_masked3 = torch.tensor(0, dtype=torch.long, device=device)
    
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}         
        
        real_wrapper = getattr(pretrain_wrapper, "module", pretrain_wrapper)
        loss1, correct1, mr_y1, mi_y1 = loss_stage1_baseline(wrapper=real_wrapper, batch=batch, pad_id=pad_id, mask_id=mask_id, vocab_size=vocab_size, evaluate=True)
        loss2, correct2, mr_y2 = loss_stage2_seed_mrna(wrapper=real_wrapper, batch=batch, pad_id=pad_id, mask_id=mask_id, vocab_size=vocab_size, evaluate=True)
        loss2_2, correct2_2, mi_y2_2 = loss_stage2_seed_mirna(wrapper=real_wrapper, batch=batch, pad_id=pad_id, mask_id=mask_id, vocab_size=vocab_size, evaluate=True)
        loss3, correct3, mr_y3, mi_y3 = loss_stage3_bispan(wrapper=real_wrapper, batch=batch, pad_id=pad_id, mask_id=mask_id, vocab_size=vocab_size, evaluate=True)
        
        # Accumulate losses
        local_loss1 += loss1
        local_loss2 += loss2
        local_loss2_2 += loss2_2
        local_loss3 += loss3
        
        # Accumulate correct predictions
        local_correct1 += correct1
        local_correct2 += correct2
        local_correct2_2 += correct2_2
        local_correct3 += correct3
        
        # Count masked tokens for each stage
        local_masked1 += mr_y1.ne(-100).sum() + mi_y1.ne(-100).sum()
        local_masked2 += mr_y2.ne(-100).sum()
        local_masked2_2 += mi_y2_2.ne(-100).sum()
        local_masked3 += mr_y3.ne(-100).sum() + mi_y3.ne(-100).sum()
    
    # Reduce across all processes
    for  t in (local_loss1, local_loss2, local_loss2_2, local_loss3, 
                local_correct1, local_correct2, local_correct2_2, local_correct3, 
                local_masked1, local_masked2, local_masked2_2, local_masked3):
        dist.all_reduce(t, op=dist.ReduceOp.SUM)

    # Convert to Python scalars
    global_loss1 = local_loss1.item()
    global_loss2 = local_loss2.item()
    global_loss2_2 = local_loss2_2.item()
    global_loss3 = local_loss3.item()
    global_correct1 = local_correct1.item()
    global_correct2 = local_correct2.item()
    global_correct2_2 = local_correct2_2.item()
    global_correct3 = local_correct3.item()
    global_masked1 = local_masked1.item()
    global_masked2 = local_masked2.item()
    global_masked2_2 = local_masked2_2.item()
    global_masked3 = local_masked3.item()
    
    avg_loss1 = global_loss1 / len(dataloader)
    avg_loss2 = global_loss2 / len(dataloader)
    avg_loss2_2 = global_loss2_2 / len(dataloader)
    avg_loss3 = global_loss3 / len(dataloader)
    
    # Calculate accuracies
    acc1 = global_correct1 / max(global_masked1, 1)  # Avoid division by zero
    acc2 = global_correct2 / max(global_masked2, 1)
    acc2_2 = global_correct2_2 / max(global_masked2_2, 1)
    acc3 = global_correct3 / max(global_masked3, 1)
    
    # Overall metrics
    total_loss = avg_loss1 + avg_loss2 + avg_loss2_2 + avg_loss3
    total_correct = global_correct1 + global_correct2 + global_correct2_2 + global_correct3
    total_masked = global_masked1 + global_masked2 + global_masked2_2 + global_masked3
    overall_acc = total_correct / max(total_masked, 1)
    
    if rank == 0:
        print(f"Evaluation Results:")
        print(f"  Stage 1 (Baseline): Loss={avg_loss1:.4f}, Accuracy={acc1:.4f}")
        print(f"  Stage 2 (Seed mRNA): Loss={avg_loss2:.4f}, Accuracy={acc2:.4f}")
        print(f"  Stage 2 (Seed miRNA): Loss={avg_loss2_2:.4f}, Accuracy={acc2_2:.4f}")
        print(f"  Stage 3 (Bispan): Loss={avg_loss3:.4f}, Accuracy={acc3:.4f}")
        print(f"  Overall: Loss={total_loss:.4f}, Accuracy={overall_acc:.4f}")
    
    return {
        'overall_loss': total_loss,
        'overall_acc': overall_acc,
        'stage1_acc': acc1,
        'stage2_acc': acc2,
        'stage2_2_acc': acc2_2,
        'stage3_acc': acc3,
    }

def run_ddp(rank, world_size, epochs, 
            accumulation_step=1, 
            mrna_max_len=520, 
            mirna_max_len=24, 
            embed_dim=256, 
            num_heads=8, 
            num_layers=4, 
            ff_dim=1024, 
            dropout_rate=0.1, 
            batch_size=32,
            lr=1e-4,
            checkpoint_path=None,
            ):
    """Run distributed training on a single process."""
    
    # Setup distributed training with synchronized random seed
    setup(rank, world_size, seed=42)

    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = f"cuda:{local_rank}"
    
    # Set device for this process
    tokenizer = CharacterTokenizer(characters=["A", "T", "C", "G", "N"],
                                model_max_length=mrna_max_len,
                                padding_side="right")
    
    train_path = os.path.join(PROJ_HOME, "TargetScan_dataset/Positive_primates_train_500_randomized_start.csv")
    valid_path = os.path.join(PROJ_HOME, "TargetScan_dataset/Positive_primates_validation_500_randomized_start.csv")

    if rank == 0:
        print(f"Loading training data from: {train_path}")
        print(f"Loading validation data from: {valid_path}")
    
    ds_train = QuestionAnswerDataset(data=load_dataset(train_path, sep=','),
                                    mrna_max_len=mrna_max_len,
                                    mirna_max_len=mirna_max_len,
                                    tokenizer=tokenizer,
                                    seed_start_col="seed start",
                                    seed_end_col="seed end",)
    
    ds_val = QuestionAnswerDataset(data=load_dataset(valid_path, sep=','),
                                  mrna_max_len=mrna_max_len,
                                  mirna_max_len=mirna_max_len,
                                  tokenizer=tokenizer,
                                  seed_start_col="seed start",
                                  seed_end_col="seed end",)
    
    # Create distributed samplers
    train_sampler = DistributedSampler(ds_train, num_replicas=world_size, rank=rank, shuffle=False)
    # Create validation sampler for distributed evaluation
    val_sampler = DistributedSampler(ds_val, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(ds_train, batch_size=batch_size, sampler=train_sampler, shuffle=False)
    val_loader = DataLoader(ds_val, batch_size=batch_size, sampler=val_sampler, shuffle=False)
    
    if rank == 0:
        print("Creating model...")
    
    model = QuestionAnsweringModel(mrna_max_len=mrna_max_len,
                                   mirna_max_len=mirna_max_len,
                                   device=device,
                                   epochs=epochs,
                                   embed_dim=embed_dim,
                                   num_heads=num_heads,
                                   num_layers=num_layers,
                                   ff_dim=ff_dim,
                                   batch_size=batch_size,
                                   lr=3e-5,
                                   seed=10020,
                                   predict_span=True,
                                   predict_binding=True,
                                   use_longformer=True)
    
    if rank == 0:
        print("Creating pretrain wrapper...")
    
    pretrain_wrapper = PairedPretrainWrapper(
                        base_model = model, 
                        vocab_size = tokenizer.vocab_size, 
                        d_model    = embed_dim,
                        embed_weight = model.predictor.sn_embedding.weight)

    # Move models to device
    model.to(device)
    pretrain_wrapper.to(device)
    pretrain_wrapper = DDP(pretrain_wrapper, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # Scale learning rate according to number of GPUs for DDP
    optimizer = AdamW(pretrain_wrapper.parameters(), lr=lr)
    
    # Setup scheduler before checkpoint loading
    steps_per_epoch = len(train_loader)
    updates_per_epoch = math.ceil(steps_per_epoch / accumulation_step)
    total_updates = epochs * updates_per_epoch
    warmup_updates = int(0.05 * total_updates)
    eta_min = 3e-5
    global_step = 0

    warmup = LinearLR(optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_updates)
    cosine = CosineAnnealingLR(optimizer, T_max=total_updates - warmup_updates, eta_min=eta_min)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_updates])
    
    # Load checkpoint if provided
    if checkpoint_path:
        if rank == 0:
            print(f"Loading checkpoint from {checkpoint_path}")
        # load model, optimizer, scheduler
        resume_data = load_training_state(ckpt_path=checkpoint_path, 
                            model=pretrain_wrapper.module.base, 
                            optimizer=optimizer, 
                            scheduler=scheduler, 
                            map_location=device)
        global_step = resume_data['global_step']

        if rank == 0:
            print(f"Resuming training from epoch {resume_data['epoch']}")
            resume_epoch = resume_data['epoch']
    else:
        resume_epoch = 0

    # Initialize wandb only on rank 0
    if rank == 0:
        wandb.login(key="your key")
        wandb.init(
            project="mirna-pretraining",
            name=f"Pretrain_DDP:{mrna_max_len}-epoch:{epochs}-gpus:{world_size}", 
            config={
                "batch_size_per_gpu": batch_size,
                "effective_batch_size": batch_size * accumulation_step * world_size,
                "accumulation_steps": accumulation_step,
                "epochs": epochs,
                "learning_rate": lr,
                "effective_learning_rate": lr,
                "world_size": world_size,
            },
            tags=["Pre-train", "MLM", "TargetScan", "DDP"],
            save_code=False,
            job_type="train",
            mode="offline",
            settings=Settings(start_method="thread")  # HPC-friendly launcher
        )
    
    start = time()
    patience = 10
    best_accuracy = 0
    count = 0  # Initialize count variable for early stopping
    start_epoch = 0
    model_checkpoints_dir = os.path.join(
        PROJ_HOME, 
        "checkpoints", 
        "TargetScan", 
        "TwoTowerTransformer",
        "Longformer", 
        str(mrna_max_len),
        "Pretrain_DDP",
    )
    
    if rank == 0:
        os.makedirs(model_checkpoints_dir, exist_ok=True)
        print(f"Checkpoint directory: {model_checkpoints_dir}")
        print(f"Using DDP training with {world_size} GPUs")
        print("Starting training loop...")
    
    for epoch in range(resume_epoch, epochs):
        # Set epoch for distributed sampler
        train_sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f"   Starting epoch {epoch+1}/{epochs}")
        
        train_loss = pretrain_loop(
            model=model,
            pretrain_wrapper=pretrain_wrapper,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            total_updates=total_updates, 
            updates_per_epoch=updates_per_epoch,
            mask_id=tokenizer.mask_token_id,
            pad_id=tokenizer.pad_token_id,
            vocab_size=tokenizer.vocab_size,
            accumulation_step=accumulation_step,
            sigma=0.5,
            rank=rank,
            global_step=global_step,
        )
        
        if rank == 0:
            print(f"Starting evaluation for epoch {epoch+1}...")
            print("Synchronizing all processes...")
        dist.barrier()

        eval_results = evaluate_mlm_ddp(
            model=model,
            pretrain_wrapper=pretrain_wrapper,
            dataloader=val_loader,
            device=device,
            vocab_size=tokenizer.vocab_size,
            pad_id=tokenizer.pad_token_id,
            mask_id=tokenizer.mask_token_id,
            rank=rank,
        )
        print("Synchronizing all processes...")
        dist.barrier()

        if rank == 0:
            # Log to wandb
            wandb.log({
                        "epoch": epoch,
                        "train/loss": train_loss["Avg Loss"],
                        "train/Token Loss": train_loss["Loss1"],
                        "train/Seed Loss": train_loss["Loss2"],
                        "train/Seed Loss mirna": train_loss["Loss2_2"],
                        "train/Bispan loss": train_loss["Loss3"],
                        "eval/loss": eval_results['overall_loss'],
                        "eval/token accuracy": eval_results['overall_acc'],
                        "eval/Token_acc": eval_results['stage1_acc'],
                        "eval/Seed_acc": eval_results['stage2_acc'],
                        "eval/Seed_acc_mirna": eval_results['stage2_2_acc'],
                        "eval/Bispan_acc": eval_results['stage3_acc'],
                    })
        
            # Update best metrics
            if eval_results['overall_acc'] > best_accuracy:
                best_accuracy = eval_results['overall_acc']
                global_step = train_loss['global_step']
                count = 0  # Reset patience counter
                
                # Save best model checkpoint
                ckpt_name = f"best_accuracy_{best_accuracy:.4f}_epoch{epoch}.pth"
                ckpt_path = os.path.join(model_checkpoints_dir, ckpt_name)
                # save the underlying model state dict (not the DDP wrapper)
                # torch.save(pretrain_wrapper.module.base.state_dict(), ckpt_path) 
                save_training_state(model=pretrain_wrapper.module.base, 
                                    optimizer=optimizer, 
                                    scheduler=scheduler, 
                                    epoch=epoch, 
                                    global_step=global_step,
                                    best_metrics={"overall_accuracy": eval_results['overall_acc']}, 
                                    ckpt_path=ckpt_path)
        
        if rank == 0:
            elapsed = time() - start
            remaining = elapsed / (epoch + 1) * (epochs - epoch - 1) / 3600
            print(f"Still remain: {remaining:.2f} hrs.")

        # Synchronize all processes
        dist.barrier()

    # Final checkpoint save
    if rank == 0:
        accuracy = eval_results['overall_acc']
        global_step = train_loss['global_step']
        ckpt_path = os.path.join(model_checkpoints_dir, f"best_accuracy_{accuracy:.4f}_epoch{epochs-1}.pth")
        bm = {"overall_accuracy": accuracy}
        save_training_state(
            model=pretrain_wrapper.module.base,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epochs-1,
            global_step=global_step,
            best_metrics=bm,
            ckpt_path=ckpt_path,
        )
        print(f"[CKPT] Final checkpoint saved")

    # Cleanup distributed training
    cleanup()

def run_single_gpu(epochs, 
                   device, 
                   accumulation_step=1, 
                   mrna_max_len=520, 
                   mirna_max_len=24, 
                   embed_dim=256, 
                   num_heads=8, 
                   num_layers=4, 
                   ff_dim=1024, 
                   dropout_rate=0.1, 
                   batch_size=32,
                   lr=1e-4):
    """Original single GPU training function."""
    tokenizer = CharacterTokenizer(characters=["A", "T", "C", "G", "N"],
                                model_max_length=mrna_max_len,
                                padding_side="right")
    
    train_path = os.path.join(PROJ_HOME, "TargetScan_dataset/Positive_primates_train_500_randomized_start.csv")
    valid_path = os.path.join(PROJ_HOME, "TargetScan_dataset/Positive_primates_validation_500_randomized_start.csv")

    print(f"Loading training data from: {train_path}")
    ds_train = QuestionAnswerDataset(data=load_dataset(train_path, sep=','),
                                    mrna_max_len=mrna_max_len,
                                    mirna_max_len=mirna_max_len,
                                    tokenizer=tokenizer,
                                    seed_start_col="seed start",
                                    seed_end_col="seed end",)
    
    print(f"Loading validation data from: {valid_path}")
    ds_val = QuestionAnswerDataset(data=load_dataset(valid_path, sep=','),
                                  mrna_max_len=mrna_max_len,
                                  mirna_max_len=mirna_max_len,
                                  tokenizer=tokenizer,
                                  seed_start_col="seed start",
                                  seed_end_col="seed end",)
    
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    
    print("Creating model...")
    model = QuestionAnsweringModel(mrna_max_len=mrna_max_len,
                                   mirna_max_len=mirna_max_len,
                                   device=device,
                                   epochs=epochs,
                                   embed_dim=embed_dim,
                                   num_heads=num_heads,
                                   num_layers=num_layers,
                                   ff_dim=ff_dim,
                                   batch_size=batch_size,
                                   lr=3e-5,
                                   seed=10020,
                                   predict_span=True,
                                   predict_binding=True,
                                   use_longformer=True)
    
    print("Creating pretrain wrapper...")
    pretrain_wrapper = PairedPretrainWrapper(
                        base_model = model, 
                        vocab_size = tokenizer.vocab_size, 
                        d_model    = embed_dim,
                        embed_weight = model.predictor.sn_embedding.weight)

    optimizer = AdamW(pretrain_wrapper.parameters(), lr=lr)
    
    # Setup scheduler before checkpoint loading
    steps_per_epoch = len(train_loader)
    updates_per_epoch = math.ceil(steps_per_epoch / accumulation_step)
    total_updates = epochs * updates_per_epoch
    warmup_updates = int(0.05 * total_updates)
    eta_min = 3e-5

    warmup = LinearLR(optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_updates)
    cosine = CosineAnnealingLR(optimizer, T_max=total_updates - warmup_updates, eta_min=eta_min)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_updates])

    # wandb.login(key="600e5cca820a9fbb7580d052801b3acfd5c92da2")
    run = wandb.init(
        project="mirna-pretraining",
        name=f"Pretrain:{mrna_max_len}-epoch:{epochs}", 
        config={
            "batch_size": batch_size * accumulation_step,
            "epochs": epochs,
            "learning_rate": lr,
        },
        tags=["Pre-train", "MLM", "Attn_reg", "TarBase+TargetScan"],
        save_code=False,
        job_type="train",
        mode="offline",
    )

    model.to(device)
    pretrain_wrapper.to(device)
    
    print("Setting up training...")
    start = time()
    patience = 10
    best_accuracy = 0
    count = 0  # Initialize count variable for early stopping
    start_epoch = 0
    global_step = 0
    model_checkpoints_dir = os.path.join(
        PROJ_HOME, 
        "checkpoints", 
        "TargetScan+TarBase", 
        "TwoTowerTransformer",
        "Longformer", 
        str(mrna_max_len),
        "Pretrain_Single",
    )
    os.makedirs(model_checkpoints_dir, exist_ok=True)
    print(f"Checkpoint directory: {model_checkpoints_dir}")


    print("Starting training loop...")
    for epoch in range(start_epoch, epochs):
        print(f"   Starting epoch {epoch+1}/{epochs}")
        train_loss = pretrain_loop(
            model=model,
            pretrain_wrapper=pretrain_wrapper,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            total_updates=total_updates, 
            updates_per_epoch=updates_per_epoch,
            mask_id=tokenizer.mask_token_id,
            pad_id=tokenizer.pad_token_id,
            vocab_size=tokenizer.vocab_size,
            accumulation_step=accumulation_step,
            sigma=0.5,  
            global_step=global_step,
            )
        
        print(f"Starting evaluation for epoch {epoch+1}...")
        eval_results = evaluate_mlm(
            model=model,
            pretrain_wrapper=pretrain_wrapper,
            dataloader=val_loader,
            device=device,
            vocab_size=tokenizer.vocab_size,
            pad_id=tokenizer.pad_token_id,
            mask_id=tokenizer.mask_token_id,
            epoch=epoch,
        )

        wandb.log({
                    "train/loss": train_loss["Avg Loss"],
                    "train/Token Loss": train_loss["Loss1"],
                    "train/Seed Loss": train_loss["Loss2"],
                    "train/Seed Loss mirna": train_loss["Loss2_2"],
                    "train/Bispan loss": train_loss["Loss3"],
                    "eval/loss": eval_results['overall_loss'],
                    "eval/token accuracy": eval_results['overall_acc'],
                    "eval/Token_acc": eval_results['stage1_acc'],
                    "eval/Seed_acc": eval_results['stage2_acc'],
                    "eval/Seed_acc_mirna": eval_results['stage2_2_acc'],
                    "eval/Bispan_acc": eval_results['stage3_acc'],
                }, step=epoch)
        
        if eval_results['overall_acc'] > best_accuracy:
            best_accuracy = eval_results['overall_acc']
            count = 0  # Reset patience counter
            
            # Save best model checkpoint
            ckpt_name = f"best_accuracy_{best_accuracy:.4f}_epoch{epoch}.pth"
            ckpt_path = os.path.join(model_checkpoints_dir, ckpt_name)
            torch.save(model.state_dict(), ckpt_path)

            model_art = wandb.Artifact(
                name="Pretrained-model",
                type="model",
                metadata={
                    "epoch": epoch,
                    "accuracy": best_accuracy
                }
            )
            model_art.add_file(ckpt_path)
            try:
                run.log_artifact(model_art, aliases=["best-pretrain"])
            except Exception as e:
                print(f"[W&B] artifact log failed at epoch {epoch}: {e}")
        else:
            count += 1
            if count >= patience:
                print("Max patience reached with no improvement. Early stopping.")
                break
        
        elapsed = time() - start
        remaining = elapsed / (epoch + 1) * (epochs - epoch - 1) / 3600
        print(f"Still remain: {remaining:.2f} hrs.")

def main():
    """Main function to launch training (single GPU or DDP)."""
    
    # Training parameters - modify these as needed
    use_ddp = True  # Set to True for DDP training, False for single GPU
    epochs = 20
    batch_size = 32
    accumulation_step = 8
    embed_dim = 1024
    ff_dim = 4096
    lr = 1e-4
    mrna_max_len = 520
    mirna_max_len = 24
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    checkpoint_path = None

    if use_ddp:
        if not torch.cuda.is_available():
            print("CUDA is not available. DDP requires CUDA.")
            return
        
        # Launch distributed training
        run_ddp(
            rank=rank,
            world_size=world_size,
            epochs=epochs,
            accumulation_step=accumulation_step,
            mrna_max_len=mrna_max_len,
            mirna_max_len=mirna_max_len,
            embed_dim=embed_dim,
            num_heads=8,
            num_layers=4,
            ff_dim=ff_dim,
            dropout_rate=0.1,
            batch_size=batch_size,
            lr=lr,
            checkpoint_path=checkpoint_path,
            )
    else:
        # Single GPU training
        if torch.backends.mps.is_available():
            device = "mps"
            print("Using MPS (Apple Silicon GPU)")
        elif torch.cuda.is_available():
            device = "cuda:1"
            print("Using CUDA GPU")
        else:
            device = "cpu"
            print("Using CPU")
        
        run_single_gpu(epochs=epochs, 
                      device=device, 
                      embed_dim=embed_dim,  
                      ff_dim=ff_dim,     
                      batch_size=batch_size, 
                      accumulation_step=accumulation_step,
                      lr=lr)

if __name__ == "__main__":
    main()  
