#!/bin/bash

# Default parameters
CONFIG_FILE=""
GPU_NUM=1
GPU_IDS="0"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG_FILE="$2"
      shift
      shift
      ;;
    --gpu_num)
      GPU_NUM="$2"
      shift
      shift
      ;;
    --gpu_ids)
      GPU_IDS="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      echo "Usage: bash train.sh --config <config file path> --gpu_num <number of GPUs> --gpu_ids <GPU ID list (comma-separated)>"
      exit 1
      ;;
  esac
done

# Check required arguments
if [ -z "$CONFIG_FILE" ]; then
  echo "Error: Configuration file path (--config) is required"
  echo "Usage: bash train.sh --config <config file path> --gpu_num <number of GPUs> --gpu_ids <GPU ID list (comma-separated)>"
  exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Error: Config file $CONFIG_FILE does not exist!"
  exit 1
fi

echo "Using config file: $CONFIG_FILE"
echo "Number of GPUs: $GPU_NUM"
echo "GPU IDs: $GPU_IDS"

# Set CUDA_VISIBLE_DEVICES environment variable
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
echo "Set CUDA_VISIBLE_DEVICES=$GPU_IDS"

# Start training
echo "Starting training..."
if [ "$GPU_NUM" -gt 1 ]; then
  # Multi-GPU training
  torchrun --standalone --nnodes=1 --nproc_per_node="$GPU_NUM" train.py --config "$CONFIG_FILE"
else
  # Single-GPU training
  python train.py --config "$CONFIG_FILE"
fi

echo "Training finished" 