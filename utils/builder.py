import torch
from torch.optim import Optimizer, Adam, AdamW, SGD
from torch.optim.lr_scheduler import LambdaLR, StepLR, CosineAnnealingLR, LinearLR, SequentialLR
import math
import sys
import os

from models.pgnet import PGNet

def build_model(config):
    """
    Builds the model based on the provided configuration.

    Args:
        config (dict): The full configuration dictionary.

    Returns:
        torch.nn.Module: The instantiated model.
    """
    model_type = config['model'].get('type', config['model']['name'])  # For backward compatibility
    model_config = config['model']
    dataset_config = config['dataset'] # Access dataset config if needed

    print(f"Building model of type: {model_type}")

    if model_type == 'PGNet':
        model = PGNet(config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


    return model


def build_optimizer(model, config):
    """
    Builds the optimizer based on the provided configuration.

    Args:
        model (torch.nn.Module): The model whose parameters need optimization.
        config (dict): The full configuration dictionary.

    Returns:
        torch.optim.Optimizer: The instantiated optimizer.
    """
    opt_config = config['training']['optimizer_params']
    optimizer_type = config['training']['optimizer'].lower()
    lr = float(config['training']['learning_rate'])
    weight_decay = opt_config.get('weight_decay', 0.01) # Default from your train.py

    decay, no_decay = [], []
    model_to_inspect = model.module if hasattr(model, 'module') else model
    for n, p in model_to_inspect.named_parameters():
        if p.requires_grad:
            # Check conditions for applying weight decay (modify if needed)
            if p.dim() > 1 and 'norm' not in n and 'bias' not in n:
                decay.append(p)
                # print(f"Param '{n}' assigned to decay group.") # Debug print
            else:
                no_decay.append(p)
                # print(f"Param '{n}' assigned to no_decay group.") # Debug print

    param_groups = [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]
    print(f"Optimizer param groups: {len(decay)} params with WD={weight_decay}, {len(no_decay)} params with WD=0.0")


    # --- Optimizer Instantiation ---
    print(f"Building optimizer: {optimizer_type}")
    if optimizer_type == 'adam':
        beta1 = opt_config.get('adam_beta1', 0.9)
        beta2 = opt_config.get('adam_beta2', 0.999)
        optimizer = Adam(param_groups, lr=lr, betas=(beta1, beta2))
        print(f"Using Adam optimizer with lr={lr}, betas=({beta1}, {beta2})")
    elif optimizer_type == 'adamw':
        beta1 = opt_config.get('adamw_beta1', 0.9)
        beta2 = opt_config.get('adamw_beta2', 0.95) # Default from your train.py
        optimizer = AdamW(param_groups, lr=lr, betas=(beta1, beta2))
        print(f"Using AdamW optimizer with lr={lr}, betas=({beta1}, {beta2})")
    elif optimizer_type == 'sgd':
        momentum = opt_config.get('sgd_momentum', 0.9)
        optimizer = SGD(param_groups, lr=lr, momentum=momentum)
        print(f"Using SGD optimizer with lr={lr}, momentum={momentum}")
    else:
        raise ValueError(f"Unsupported optimizer type specified in config: {optimizer_type}")

    return optimizer


def build_scheduler(optimizer, config, train_loader_len=None):
    """
    Builds the learning rate scheduler based on the provided configuration.
    Can handle both epoch-based and step-based schedules.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer instance.
        config (dict): The full configuration dictionary.
        train_loader_len (int, optional): The number of batches in the training loader (steps per epoch).
                                          Required for epoch-based scheduling.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: The instantiated scheduler.
    """
    scheduler_type = config['training']['lr_scheduler'].lower()
    
    print(f"Building scheduler: {scheduler_type}")

    use_step_based_schedule = 'max_steps' in config['training']

    if scheduler_type == 'step':
        if use_step_based_schedule:
             raise ValueError("StepLR scheduler is epoch-based and not compatible with max_steps training. Use a different scheduler.")
        assert train_loader_len is not None, "train_loader_len is required for StepLR scheduler."
        step_size = config['training']['lr_step_size']
        gamma = config['training']['lr_gamma']
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        print(f"Using StepLR scheduler with step_size={step_size}, gamma={gamma}")

    elif scheduler_type == 'cosine':
        num_warmup_steps = 0
        num_main_cosine_steps = 1 # Default to 1 to avoid errors

        if use_step_based_schedule:
            # Step-based cosine schedule with warmup
            max_steps = int(config['training']['max_steps'])
            warmup_steps = int(config['training'].get('warmup_steps', 0))
            
            assert warmup_steps >= 0, "warmup_steps cannot be negative."
            if warmup_steps > 0:
                assert warmup_steps < max_steps, f"warmup_steps ({warmup_steps}) must be less than max_steps ({max_steps})."
            
            num_warmup_steps = warmup_steps
            num_main_cosine_steps = max(1, max_steps - num_warmup_steps)
            print(f"Scheduler mode: Step-based (max_steps: {max_steps}, warmup_steps: {warmup_steps})")
        
        else:
            # Epoch-based cosine schedule with warmup (original logic)
            assert train_loader_len is not None, "train_loader_len is required for epoch-based cosine scheduler."
            max_epoch = int(config['training']['max_epoch'])
            warmup_epochs = config['training'].get('warmup_epochs', 0)
            
            assert warmup_epochs >= 0, "warmup_epochs cannot be negative."
            if warmup_epochs > 0:
                 assert warmup_epochs < max_epoch, f"warmup_epochs ({warmup_epochs}) must be less than max_epoch ({max_epoch})."

            num_warmup_steps = warmup_epochs * train_loader_len
            num_total_training_steps = max_epoch * train_loader_len
            num_main_cosine_steps = max(1, num_total_training_steps - num_warmup_steps)
            print(f"Scheduler mode: Epoch-based (max_epoch: {max_epoch}, warmup_epochs: {warmup_epochs})")

        # scheduler1 for linear warmup
        warmup_start_factor = float(config['training'].get('warmup_start_factor', 0.1))
        scheduler1 = LinearLR(optimizer, start_factor=warmup_start_factor, end_factor=1.0, total_iters=num_warmup_steps)

        # scheduler2 for cosine decay
        cosine_eta_min = float(config['training'].get('cosine_eta_min', 1e-5))
        scheduler2 = CosineAnnealingLR(optimizer, T_max=num_main_cosine_steps, eta_min=cosine_eta_min)

        scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[num_warmup_steps])
        print(f"Using SequentialLR: Linear Warmup for {num_warmup_steps} steps then "
                f"Cosine Annealing for {num_main_cosine_steps} steps (eta_min={cosine_eta_min}).")
    else:
        raise ValueError(f"Unsupported scheduler type specified in config: {scheduler_type}")

    return scheduler 