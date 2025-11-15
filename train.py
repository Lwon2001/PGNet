# coding=utf-8
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-C', '--config', type=str, default='configs/default.yaml',
                    help='Path to the YAML config file')
parser.add_argument('--local_rank', type=int, default=-1, help='Local process rank for distributed training')
args = parser.parse_args()

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from metrics.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from pointnet2_ops.pointnet2_utils import furthest_point_sample
from torch.utils.data import DataLoader
from utils.dataloader import *
import torch
import torch.nn as nn
from metrics import meter
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import kaolin as kal 
from tqdm import tqdm
import torch.optim as optim
import numpy as np
from datetime import datetime
import random
from torch.utils.data import Subset
from torch.optim.lr_scheduler import LambdaLR
from utils.config_utils import load_config, setup_output_dirs
from utils.builder import build_model, build_optimizer, build_scheduler

# Load configuration
config = load_config(args.config)

# Set NCCL timeout (e.g., 1 hour = 3600 seconds)
nccl_timeout_sec = config.get('training', {}).get('nccl_timeout', 3600) # Read from config, default 3600
os.environ['NCCL_TIMEOUT'] = str(nccl_timeout_sec) 
if int(os.environ.get("RANK", 0)) == 0: # Print only once in the main process (use env RANK because global RANK may not be set yet)
    print(f"Set NCCL timeout to: {nccl_timeout_sec} seconds")

# Check if we're in distributed mode
DISTRIBUTED_MODE = False
WORLD_SIZE = 1
RANK = 0

# Check for distributed training via torchrun/pytorch launcher
if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
    DISTRIBUTED_MODE = True
    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    RANK = int(os.environ.get("RANK", 0))
# Check for distributed training via command line args
elif args.local_rank >= 0:
    DISTRIBUTED_MODE = True
    # If local_rank is set but no WORLD_SIZE, assume single-node multi-GPU
    WORLD_SIZE = torch.cuda.device_count()
    RANK = args.local_rank

# Set visible devices if not in distributed mode
if not DISTRIBUTED_MODE:
    gpu_id = config.get('hardware', {}).get('gpu', '0')
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id



# --- Get parameters from config (step-based) ---
# Dataset configuration
CLASS = config['dataset']['category']
DATA_PATH = config['dataset']['data_path']
SAMPLING_METHOD = config['dataset']['sampling_method']
NUM_GEN_POINTS = int(config['dataset']['gen_points'])
NORMALIZE_GEN = config['dataset'].get('normalize_gen', True)
DEBUG_SUBSET_RATIO = float(config['dataset'].get('debug_subset_ratio', 1.0))

# Hardware configuration
NUM_WORKERS = int(config.get('hardware', {}).get('num_workers', 4))

# Training configuration
LR = float(config['training']['learning_rate'])

# --- Automatically compute batch size and gradient accumulation steps ---
# Read experimental intent from config
global_batch_size = int(config['training']['global_batch_size'])
GRAD_ACCUM_STEPS = int(config.get('training', {}).get('gradient_accumulation_steps', 1))

# Validate parameter consistency
if global_batch_size % (GRAD_ACCUM_STEPS * WORLD_SIZE) != 0:
    raise ValueError(
        f"Global batch size ({global_batch_size}) must be divisible by "
        f"gradient accumulation steps ({GRAD_ACCUM_STEPS}) * number of GPUs ({WORLD_SIZE})."
    )

# Compute per-GPU batch size for a single forward pass
BATCH_SIZE = global_batch_size // (GRAD_ACCUM_STEPS * WORLD_SIZE)

if BATCH_SIZE < 1:
     raise ValueError(
        f"Computed per-GPU batch size ({BATCH_SIZE}) is less than 1. Please adjust global_batch_size or "
        f"gradient_accumulation_steps."
    )

if RANK == 0:
    print("-" * 40)
    print("Batch size and gradient accumulation configuration:")
    print(f"  - Target logical global batch size: {global_batch_size}")
    print(f"  - Gradient accumulation steps: {GRAD_ACCUM_STEPS}")
    print(f"  - Computed per-GPU physical batch size: {BATCH_SIZE}")
    print("-" * 40)

# Step-based training parameters
MAX_STEPS = int(config['training']['max_steps'])
EVAL_STEPS = int(config['training']['eval_steps'])
# CHECKPOINT_STEPS = int(config['training'].get('checkpoint_steps', EVAL_STEPS)) # Default: synchronized with evaluation interval


def save_checkpoint(step, model, optimizer=None, scheduler=None, is_best=False, metrics=None, best_val_metric=float('inf'), checkpoint_dir=None):
    """
    Save model checkpoint (step-based).
    
    Args:
        step (int): Current step.
        model (nn.Module): Model to save.
        optimizer, scheduler: Optimizer and scheduler objects.
        is_best (bool): True if this is the best model so far.
        metrics (dict): Metrics dict for the *current step*.
        best_val_metric (float): Best validation metric *so far*.
        checkpoint_dir (str): Directory to save checkpoints.
    """
    if checkpoint_dir is None:
        if RANK == 0:
            print("Warning: checkpoint_dir is None, cannot save checkpoint.")
        return

    metrics_str = ""
    if metrics:
        # Convert keys like 'val/loss_cd...' to 'val_loss_cd...'
        metrics_str = "_".join([f"{k.replace('/', '_')}_{v:.6f}" for k, v in metrics.items()])
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'best_val_metric': best_val_metric,
        'config': config # Save config for reference
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
    try:
        # Remove old latest checkpoint files
        for f in os.listdir(checkpoint_dir):
            if f.startswith("latest_step_"):
                os.remove(os.path.join(checkpoint_dir, f))
        
        # Save latest checkpoint
        latest_filename = f"latest_step_{step}{'_' + metrics_str if metrics_str else ''}.pth"
        latest_path = os.path.join(checkpoint_dir, latest_filename)
        torch.save(checkpoint, latest_path)

        # Save checkpoints at intervals (optional)
        # step_filename = f"step_{step}{'_' + metrics_str if metrics_str else ''}.pth"
        # step_path = os.path.join(checkpoint_dir, step_filename)
        # torch.save(checkpoint, step_path)

        if is_best:
            # Remove old best checkpoint files
            for f in os.listdir(checkpoint_dir):
                if f.startswith("best_step_"):
                    os.remove(os.path.join(checkpoint_dir, f))
            
            best_filename = f"best_step_{step}{'_' + metrics_str if metrics_str else ''}.pth"
            best_path = os.path.join(checkpoint_dir, best_filename)
            torch.save(checkpoint, best_path)
            print(f"\nSaving best model: {best_filename} (metric: {best_val_metric:.6f})\n")
            
    except Exception as e:
        print(f"Error saving checkpoint: {str(e)}")


def train_step(data, model, criterion_cd_l1):
    """
    Execute a single forward pass and loss computation.
    Does not perform optimizer steps; returns loss tensor for gradient accumulation.
    """
    model.train()
    
    task_ids, images, pcs_gt_cpu, pcs_gen_cpu, pc_parts_cpu = data
    device = next(model.parameters()).device
    
    pcs_gt = pcs_gt_cpu.to(device, non_blocking=True)
    pcs_gen = pcs_gen_cpu.to(device, non_blocking=True)
    pc_parts = pc_parts_cpu.to(device, non_blocking=True)

    batch_size = pcs_gt.size(0)
    
    loss_weights = config.get('training', {}).get('loss_weights', {})
    
    model_output = model(pcs_gen, pc_parts)
    all_pcds = model_output["all_pcds"]
    
    # Sample GT point cloud for computing loss
    num_fine_points = config['dataset'].get('fine_points', 2048)
    sampled_indices = furthest_point_sample(pcs_gt, num_fine_points)
    
    if sampled_indices is None:
        if RANK == 0:
            print(f"Warning: furthest_point_sample returned None during training, skipping this physical batch.")
        return None

    indices = sampled_indices.unsqueeze(-1).expand(batch_size, num_fine_points, 3).to(device)
    indices = indices.long()
    gt = torch.gather(pcs_gt, 1, indices)
    
    total_loss = 0.
    train_losses = {}
    assert len(all_pcds) == len(loss_weights)
    for i, pcd in enumerate(all_pcds):
       
        loss_weight = loss_weights[i] if i < len(loss_weights) else 1.0
        loss_cd = criterion_cd_l1(pcd, gt)
        total_loss += loss_weight * loss_cd
        
        # Naming convention
        if i == 0:
            loss_name = 'train/loss_cd_l1_coarse'
        elif i == len(all_pcds) - 1:
            loss_name = 'train/loss_cd_l1_fine'
        else:
            loss_name = f'train/loss_cd_l1_intermediate_{i}'
        train_losses[loss_name] = loss_cd.item()

    train_losses['train/loss_all'] = total_loss
    
    return train_losses


def validate(model, test_loader, criterion_cd_l2):
    """
    Validation loop - using L2 Chamfer Distance.
    """
    model.eval()
    losses_cd_l2_fine_meter = meter.AverageValueMeter()
    losses_cd_l2_coarse_meter = meter.AverageValueMeter()
    intermediate_losses_meters = {}

    rank = dist.get_rank() if DISTRIBUTED_MODE else 0
    world_size = dist.get_world_size() if DISTRIBUTED_MODE else 1
    
    pbar = None
    if rank == 0:
        pbar = tqdm(total=len(test_loader), desc="Validating")
    
    num_fine_points = config['dataset'].get('fine_points', 2048)
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            task_ids, images, pcs_gt, pcs_gen, pc_parts = data
            
            device = next(model.parameters()).device
            pcs_gt = pcs_gt.to(device, non_blocking=True)
            pcs_gen = pcs_gen.to(device, non_blocking=True)
            pc_parts = pc_parts.to(device, non_blocking=True)
            
            batch_size = pcs_gt.size(0)
            
            model_output = model(pcs_gen, pc_parts)
            all_pcds = model_output["all_pcds"]
            
            # Initialize intermediate_losses_meters on the first iteration
            if i == 0 and len(all_pcds) > 2:
                for j in range(1, len(all_pcds) - 1):
                    intermediate_losses_meters[f'val/loss_cd_l2_intermediate_{j}'] = meter.AverageValueMeter()

            sampled_indices = furthest_point_sample(pcs_gt, num_fine_points)
            if sampled_indices is None:
                if rank == 0:
                    print(f"Warning: furthest_point_sample returned None during validation, skipping this batch.")
                if pbar:
                    pbar.update(1)
                continue

            indices = sampled_indices.unsqueeze(-1).expand(batch_size, num_fine_points, 3).to(device)
            indices = indices.long()
            gt = torch.gather(pcs_gt, 1, indices)
            
            loss_cd_l2_coarse = criterion_cd_l2(all_pcds[0], gt)
            loss_cd_l2_fine = criterion_cd_l2(all_pcds[-1], gt)
            
            # Collect loss values for synchronization
            loss_values = [loss_cd_l2_coarse.item(), loss_cd_l2_fine.item()]
            for j in range(1, len(all_pcds) - 1):
                loss_inter = criterion_cd_l2(all_pcds[j], gt)
                loss_values.append(loss_inter.item())
            
            loss_tensor = torch.tensor(loss_values, device=device)
            
            if DISTRIBUTED_MODE:
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                loss_tensor /= world_size
            
            # Only the main process updates the meters
            if rank == 0:
                losses_cd_l2_coarse_meter.add(loss_tensor[0].item())
                losses_cd_l2_fine_meter.add(loss_tensor[1].item())
                for j in range(len(intermediate_losses_meters)):
                    meter_key = f'val/loss_cd_l2_intermediate_{j+1}'
                    intermediate_losses_meters[meter_key].add(loss_tensor[2+j].item())
                if pbar:
                    pbar.update(1)

    if pbar:
        pbar.close()
    
    if DISTRIBUTED_MODE:
        dist.barrier()

    val_losses = {}
    if rank == 0:
        val_losses = {
            'val/loss_cd_l2_coarse': losses_cd_l2_coarse_meter.value()[0],
            'val/loss_cd_l2_fine': losses_cd_l2_fine_meter.value()[0]
        }
        for k, v in intermediate_losses_meters.items():
            val_losses[k] = v.value()[0]
    
    return val_losses


def setup_distributed(rank, world_size):
    """
    Set up distributed training environment.
    """
    if "MASTER_ADDR" not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if "MASTER_PORT" not in os.environ:
        os.environ['MASTER_PORT'] = '12355'
    
    if "WORLD_SIZE" in os.environ and "RANK" in os.environ:
        dist.init_process_group(backend='nccl')
    else:
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:12355', world_size=world_size, rank=rank)
    
    torch.cuda.set_device(rank)
    dist.barrier()

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def main_worker(rank, world_size, args, config):
    global RANK, WORLD_SIZE, DISTRIBUTED_MODE
    
    RANK = rank
    WORLD_SIZE = world_size
    
    seed = 42 + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if DISTRIBUTED_MODE:
        setup_distributed(rank, world_size)
        if rank == 0: print(f"Distributed environment initialized, world size: {world_size}")
    
    if torch.cuda.is_available():
        cudnn.benchmark = True
        if rank == 0: print("CUDNN benchmark enabled")
    
    # --- Directory and path setup ---
    output_path, checkpoint_dir, log_dir = None, None, None
    if rank == 0:
        paths = setup_output_dirs(config)
        output_path = paths['base']
        print(f"New training run: creating experiment directory {output_path}")

        if output_path:
            checkpoint_dir = os.path.join(output_path, config['output']['checkpoint_dir'])
            log_dir = os.path.join(output_path, config['output']['log_dir'])
            os.makedirs(checkpoint_dir, exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)
    
    # Synchronize paths in distributed mode
    if DISTRIBUTED_MODE:
        obj_list = [output_path]
        dist.broadcast_object_list(obj_list, src=0)
        if rank != 0: output_path = obj_list[0]
        
        if output_path:
            checkpoint_dir = os.path.join(output_path, config['output']['checkpoint_dir'])
            log_dir = os.path.join(output_path, config['output']['log_dir'])
            if rank != 0:
                os.makedirs(checkpoint_dir, exist_ok=True)
                os.makedirs(log_dir, exist_ok=True)
        dist.barrier()
        
    # --- Print training info ---
    if rank == 0:
        print("\n--- Training configuration ---")
        print(f"Category: {CLASS}, Model: {config['model']['name']}")
        print(f"Learning rate: {LR}")
        print(f"Max steps: {MAX_STEPS}, Eval interval: {EVAL_STEPS}")
        

        print(f"Distributed training: {DISTRIBUTED_MODE}, World size: {world_size}")
        print("----------------\n")

    # --- Data loaders ---
    train_dataset = PCDataLoader(
        filepath=DATA_PATH + '/train_list.txt',
        data_path=DATA_PATH, 
        status='train', 
        view_align=False, 
        category=CLASS, 
        sampling_method=SAMPLING_METHOD, 
        num_points=NUM_GEN_POINTS, 
        normalize_gen=NORMALIZE_GEN
    )
    test_dataset = PCDataLoader(
        filepath=DATA_PATH + '/test_list.txt', 
        data_path=DATA_PATH, 
        status='test', 
        view_align=False, 
        category=CLASS, 
        sampling_method=SAMPLING_METHOD, 
        num_points=NUM_GEN_POINTS, 
        normalize_gen=NORMALIZE_GEN
    )
    
    if DEBUG_SUBSET_RATIO < 1.0:
        random.seed(42)
        train_subset_size = max(1, int(len(train_dataset) * DEBUG_SUBSET_RATIO))
        train_indices = random.sample(range(len(train_dataset)), train_subset_size)
        train_dataset = Subset(train_dataset, train_indices)
        
        test_subset_size = max(1, int(len(test_dataset) * DEBUG_SUBSET_RATIO))
        test_indices = random.sample(range(len(test_dataset)), test_subset_size)
        test_dataset = Subset(test_dataset, test_indices)
        if rank == 0: print(f"DEBUG mode: using {DEBUG_SUBSET_RATIO:.2%} of the dataset")

    if rank == 0:
        print(f"Train set size: {len(train_dataset)}, Test set size: {len(test_dataset)}")

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, drop_last=True) if DISTRIBUTED_MODE else None
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False) if DISTRIBUTED_MODE else None
    
    train_loader = DataLoader(train_dataset, BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=(train_sampler is None), drop_last=True, collate_fn=custom_collate, sampler=train_sampler, pin_memory=True)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False, drop_last=False, collate_fn=custom_collate, sampler=test_sampler, pin_memory=True)

    # --- Model, optimizer, scheduler ---
    model = build_model(config)
    device = torch.device(f'cuda:{rank}')
    model = model.to(device)

    if DISTRIBUTED_MODE:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    
    g_optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(g_optimizer, config)
    
    # --- Initialize training state ---
    current_step = 0
    best_val_metric = float('inf')
    # early_stopping_counter = 0
            
    # --- Training preparation ---
    criterion_cd_l1 = ChamferDistanceL1().to(device)
    criterion_cd_l2 = ChamferDistanceL2().to(device)
    board_writer = SummaryWriter(log_dir) if rank == 0 else None
    
    # Infinite data loader
    def infinite_loader(loader, sampler, start_epoch=0):
        epoch = start_epoch
        while True:
            if sampler:
                sampler.set_epoch(epoch)
            for data in loader:
                yield data
            epoch += 1
    
    # Compute starting epoch (currently starts from 0)
    start_epoch = current_step // len(train_loader)
    data_iterator = iter(infinite_loader(train_loader, train_sampler, start_epoch))

    pbar = None
    if rank == 0:
        pbar = tqdm(initial=current_step, total=MAX_STEPS, desc="Training")

    # --- Training loop ---
    grad_clip_enabled = config.get('training', {}).get('grad_clip', {}).get('enabled', False)
    max_norm = config.get('training', {}).get('grad_clip', {}).get('max_norm', 1.0)

    while current_step < MAX_STEPS:
        
        g_optimizer.zero_grad()

        accumulated_losses = {
            'train/loss_all': 0.0,
            'train/loss_cd_l1_fine': 0.0,
            'train/loss_cd_l1_coarse': 0.0
        }

        # --- Gradient accumulation loop ---
        for i in range(GRAD_ACCUM_STEPS):
            data = next(data_iterator)
            
            # 1. Compute loss (no backward yet)
            loss_dict = train_step(data, model, criterion_cd_l2)

            if loss_dict is None:
                continue # If sampling fails, skip this physical batch

            loss = loss_dict['train/loss_all']
            
            # 2. Normalize loss and backpropagate
            normalized_loss = loss / GRAD_ACCUM_STEPS
            normalized_loss.backward()

            # 3. Accumulate loss values for logging
            accumulated_losses['train/loss_all'] += loss.item() / GRAD_ACCUM_STEPS
            accumulated_losses['train/loss_cd_l1_fine'] += loss_dict['train/loss_cd_l1_fine'] / GRAD_ACCUM_STEPS
            accumulated_losses['train/loss_cd_l1_coarse'] += loss_dict['train/loss_cd_l1_coarse'] / GRAD_ACCUM_STEPS
        
        # --- After accumulation, perform optimization step ---
        
        # 4. Gradient clipping (before optimizer.step)
        if grad_clip_enabled:
            params_to_clip = [p for p in model.parameters() if p.grad is not None]
            if params_to_clip:
                torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm)
        
        # 5. Update weights
        g_optimizer.step()
        
        # 6. Update learning rate (once per logical step)
        scheduler.step()

        # --- Logging, validation, checkpoint saving ---
        if rank == 0:
            if pbar:
                pbar.update(1)
                pbar.set_postfix({k.split('/')[-1]: f"{v:.4f}" for k, v in accumulated_losses.items()})
            if board_writer:
                for k, v in accumulated_losses.items():
                    board_writer.add_scalar(k, v, global_step=current_step)
                board_writer.add_scalar('lr', g_optimizer.param_groups[0]['lr'], global_step=current_step)

        if current_step > 0 and (current_step % EVAL_STEPS == 0 or current_step == MAX_STEPS -1) :
            val_losses = validate(model, test_loader, criterion_cd_l2)
            
            # Only the main process handles validation results
            if rank == 0:
                print(f"\n--- Step {current_step} validation results ---")
                for k, v in val_losses.items():
                    print(f"{k}: {v:.6f}")
                print("--------------------------\n")

                if board_writer:
                    for k, v in val_losses.items():
                        board_writer.add_scalar(k, v, global_step=current_step)
                
                # current_val_metric = val_losses.get(EARLY_STOPPING_METRIC)
                current_val_metric = val_losses.get('val/loss_cd_l2_fine') # Use fine loss for best model
                is_best = False
                if current_val_metric is not None and current_val_metric < best_val_metric:
                    best_val_metric = current_val_metric
                    is_best = True

                # Save checkpoint
                save_checkpoint(
                    step=current_step,
                    model=model.module if DISTRIBUTED_MODE else model,
                    optimizer=g_optimizer,
                    scheduler=scheduler,
                    is_best=is_best,
                    metrics=val_losses,
                    best_val_metric=best_val_metric,
                    checkpoint_dir=checkpoint_dir
                )


        current_step += 1
        
        if DISTRIBUTED_MODE:
            dist.barrier()

    if pbar: pbar.close()

    if rank == 0:
        print("\nTraining finished!")
        
        print(f"Best validation metric (val/loss_cd_l2_fine): {best_val_metric:.6f}")
    
    if DISTRIBUTED_MODE:
        dist.barrier()

def main():
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        main_worker(local_rank, world_size, args, config)
    elif args.local_rank >= 0:
        world_size = torch.cuda.device_count()
        main_worker(args.local_rank, world_size, args, config)
    else:
        main_worker(0, 1, args, config)

if __name__ == "__main__":
    main()
    if DISTRIBUTED_MODE and dist.is_initialized():
        cleanup_distributed()