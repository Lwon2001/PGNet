#!/usr/bin/env python
# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 SYSU/Wang Luo
import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
# Import configuration utilities
from utils.config_utils import load_config
from utils.dataloader import PCDataLoader, custom_collate
from pointnet2_ops.pointnet2_utils import furthest_point_sample
# Import build_model function from builder
from utils.builder import build_model
# Import Chamfer Distance
from metrics.chamfer_dist import ChamferDistanceL2
# Import the new F1Score module
from metrics.F_Score.f_score import F1Score
# Import EMD module
from metrics.EMD.emd_module import emdModule

def parse_args():
    """
    Parse command-line arguments.

    Only keep a few arguments needed for category-level evaluation:
    - Path to the config file (defines category and other data configurations)
    - Path to model weights
    - Device
    """
    parser = argparse.ArgumentParser(description='MMPC model evaluation (category-level, no visualization)')
    parser.add_argument(
        '-C', '--config', type=str,
        default='configs/ShapeNet-ViPC/MMPC_plane.yaml',
        help='Path to the YAML config file (specifies dataset category and other settings)'
    )
    parser.add_argument(
        '-M', '--model', type=str, required=True,
        help='Path to the pretrained model weights (.pth file)'
    )
    parser.add_argument(
        '--device', type=str, default='cuda:0',
        help='Device for inference, e.g. "cuda:0" or "cpu"'
    )
    return parser.parse_args()


def load_model(model_path, config, device):
    """
    Load a pretrained model based on configuration.
    
    Args:
        model_path (str): Path to the model checkpoint
        config (dict): Model configuration
        device (str): Device to load the model on
        
    Returns:
        model (nn.Module): Loaded model
    """
    # Build model from config using builder.build_model
    model = build_model(config).to(device)
    
    # Load checkpoint
    print(f"Loading weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model successfully loaded from {model_path}")
    if 'step' in checkpoint:
        print(f"Training step: {checkpoint['step']}")
    if 'metrics' in checkpoint and checkpoint['metrics']:
        metrics_str = ", ".join([f"{k}: {v:.6f}" for k, v in checkpoint['metrics'].items()])
        print(f"Validation metrics: {metrics_str}")
    return model

def main():
    """
    Main function: evaluate the full test set for the category specified in the config file.
    Each evaluation outputs the average of 3 metrics:
    - L2 Chamfer Distance (Fine result)
    - F-Score (th = 0.001)
    - EMD
    """
    args = parse_args()
    
    # Load config (category and other info are read from YAML)
    config = load_config(args.config)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Read dataset-related parameters from config
    sampling_method = config['dataset']['sampling_method']
    num_points = config['dataset'].get('num_points', config['dataset'].get('gen_points', 2048))
    category = config['dataset']['category']
    data_path = config['dataset']['data_path']
    normalize_gen = config['dataset'].get('normalize_gen', True)
    
    # Infer generated point cloud path (keep consistent with PCDataLoader logic)
    gen_root = os.path.join(data_path, 'ShapeNetViPC-Gen')
    trellis_gen_path = os.path.join(gen_root, 'trellis', sampling_method, f'num_points_{num_points}')
    if os.path.exists(trellis_gen_path):
        gen_path_for_print = trellis_gen_path
    else:
        raise FileNotFoundError(f"Trellis gen point cloud path not found: {trellis_gen_path}")
    
    # Print current evaluation config
    print("Inference / evaluation config:")
    print(f"  Category: {category}")
    print(f"  Sampling method: {sampling_method}")
    print(f"  Num points: {num_points}")
    print(f"  Model type: {config['model']['name']}")
    print(f"  Generated point cloud path: {gen_path_for_print}")
    print(f"  Device: {device}")
    
    # Load model
    model = load_model(args.model, config, device)
    
    # Define evaluation metrics
    criterion_cd_l2 = ChamferDistanceL2().to(device)  # L2 Chamfer Distance
    criterion_f1 = F1Score().to(device)               # F-Score
    criterion_emd = emdModule().to(device)            # EMD
    
    
    # Build test dataset
    test_dataset = PCDataLoader(
        data_path + '/test_list.txt',
        data_path=data_path,
        status='test',
        view_align=False,
        category=category,
        sampling_method=sampling_method,
        num_points=num_points,
        normalize_gen=normalize_gen,
    )
    
    print(f"Loaded test dataset with {len(test_dataset)} samples")

    # DataLoader
    inference_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate,
    )
    
    # Store metrics for all samples
    all_cd_l2 = []
    all_f_scores = []
    all_emds = []
    
    print(f"\nEvaluating on the entire test set ({len(inference_loader)} batches)...")
    
    for data_batch in tqdm(inference_loader, desc="Evaluating all samples"):
        # The return format of collate_fn is consistent with training
        task_id_batch, image_batch, pc_gt_batch, pc_gen_batch, pc_part_batch = data_batch
        
        with torch.no_grad():
            # Prepare inputs
            partial_pc = pc_part_batch.to(device)   # (1, Np, 3)
            gen_pc = pc_gen_batch.to(device)        # (1, Ng, 3)
            gt_from_loader = pc_gt_batch.to(device) # (1, Ng, 3)
            
            # Forward inference
            model_output = model(gen_pc, partial_pc)
            all_pcds = model_output["all_pcds"]
            fine = all_pcds[-1]  # Use only Fine result for evaluation
            
            # Use FPS to sample 2048 points from GT
            indices = furthest_point_sample(gt_from_loader, 2048).unsqueeze(-1).expand(1, 2048, 3)
            indices = indices.long()
            gt = torch.gather(gt_from_loader, 1, indices)
            
            # 1) L2 Chamfer Distance (Fine)
            loss_cd_fine_l2 = criterion_cd_l2(fine, gt)
            all_cd_l2.append(loss_cd_fine_l2.item())
            
            # 2) F-Score (th = 0.001)
            fscore_val, _, _ = criterion_f1(fine, gt, threshold=0.001)
            all_f_scores.append(fscore_val.item())
            
            # 3) EMD (take sqrt then * 1000 to stay consistent with the original code)
            emd_dist, _ = criterion_emd(fine, gt, eps=0.005, iters=100)
            emd_val = torch.sqrt(emd_dist).mean().item() * 1000.0
            all_emds.append(emd_val)
    
    # Compute average metrics
    avg_cd_l2 = np.mean(all_cd_l2) if all_cd_l2 else float('nan')
    avg_f_score = np.mean(all_f_scores) if all_f_scores else float('nan')
    avg_emd = np.mean(all_emds) if all_emds else float('nan')
    
    print("\n--- Evaluation finished ---")
    print(f"Average L2 Chamfer Distance (Fine): {avg_cd_l2 * 1000:.4f} * 10^-3")
    print(f"Average F-Score (th=0.001): {avg_f_score:.6f}")
    print(f"Average EMD: {avg_emd:.6f}")


if __name__ == "__main__":
    main() 