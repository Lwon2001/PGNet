<h2 align="center">Rethinking Multimodal Point Cloud Completion: <br/>A Completion-by-Correction Perspective</h2>
<p align="center">
  <!-- arXiv paper -->
  <a href="https://arxiv.org/abs/2511.12170">
    <img src="https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white" alt="arXiv">
  </a>
  <!-- Hugging Face: dataset -->
  <a href="https://huggingface.co/datasets/Wang131/ShapeNetViPC-Gen">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue" alt="Hugging Face Dataset">
  </a>
  <!-- Hugging Face: checkpoints -->
  <a href="https://huggingface.co/datasets/Wang131/PGNet_ckpt">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoints-orange" alt="Hugging Face Checkpoints">
  </a>
</p>

## Overview

This repository contains the official implementation for "Rethinking Multimodal Point Cloud Completion: A Completion-by-Correction Perspective" (AAAI 2026), which introduces PGNet — a multimodal point cloud completion framework that shifts from the traditional Completion-by-Inpainting paradigm to a more robust Completion-by-Correction strategy. Instead of synthesizing missing geometry from fused features, PGNet starts with a topologically complete generative prior (via an image-to-3D model) and corrects it using partial point cloud observations. By grounding a complete scaffold with reliable geometric cues, PGNet achieves state-of-the-art performance on ShapeNet-ViPC with significantly improved structural consistency and geometric fidelity.

The main components of this repo include:

- `generate_point_cloud.py`: generate high-quality prior point clouds from rendered views using Trellis;
- `train.py` / `train.sh`: train MMPC models on ShapeNetViPC;
- `inference.py`: perform category-level evaluation on the test set (Chamfer-L2 / F-Score / EMD);
- `utils`, `metrics`, `models`, `extensions`: data loading, evaluation metrics, network architectures and CUDA extensions.

## Environment

We test our code in a **Ubuntu 24.04 LTS + NVIDIA RTX 4090 GPU** environment.

- Python 3.10
- PyTorch 2.4.0 (with CUDA)
- CUDA 12.1 (via `pytorch-cuda=12.1`)

### Recommended: create from `environment.yml`

We provide a Conda environment file:

```bash
conda env create -f environment.yml
conda activate pgnet
```


### Build CUDA Extensions

This project depends on several CUDA extensions.  
Build and install them as follows (run each block from the repo root):

```bash
# PointNet++ operators
cd extensions/pointnet2_ops_lib
pip install .

# Vox2Seq operators
cd extensions/vox2seq
pip install .

# Chamfer Distance
cd metrics/chamfer_dist
pip install .

# EMD
cd metrics/EMD
pip install .
```

## Data Preparation

We train and evaluate on the **ShapeNetViPC** dataset.  
Assume the dataset root is:

```text
./data/ShapeNetViPC-Dataset
├── ShapeNetViPC-Gen              # generated point clouds from image-to-3D model (.pt)
├── ShapeNetViPC-Partial          # partial point clouds (.dat)
├── ShapeNetViPC-GT               # complete GT point clouds (.dat)
├── ShapeNetViPC-View             # rendered views (png + metadata)
├── train_list.txt                # train split file list
└── test_list.txt                 # test  split file list
```

**`ShapeNetViPC-Gen`** contains prior point clouds generated in **this work** by applying an image-to-3D model to the rendered views in `ShapeNetViPC-View`. We also provide our pre-generated prior point clouds using Trellis at our Hugging Face dataset [`Wang131/ShapeNetViPC-Gen`](https://huggingface.co/datasets/Wang131/ShapeNetViPC-Gen).

**`ShapeNetViPC-Partial`**, **`ShapeNetViPC-GT`**, **`ShapeNetViPC-View`**,  
as well as **`train_list.txt`** and **`test_list.txt`** are directly taken from the
official **ShapeNetViPC** dataset. Please refer to the official ShapeNetViPC repo ([Hydrogenion/ViPC](https://github.com/Hydrogenion/ViPC)) for obtaining the original data.

## Generate Prior Point Clouds(Trellis)

```text
/path/to/ShapeNetViPC-Dataset/ShapeNetViPC-Gen/trellis/<sampling_method>/num_points_<N>/
```

You can generate them in batch with `generate_point_cloud.py`:

```bash
python generate_point_cloud.py \
  --data_path /path/to/ShapeNetViPC-Dataset \
  --output_dir /path/to/ShapeNetViPC-Dataset/ShapeNetViPC-Gen/trellis \
  --categories plane,chair,table \
  --gpu_ids 0,1 \
  --num_workers_per_gpu 1 \
  --prefetch_size 20 \
  --loader_threads 5 \
  --sampling_threads 10 \
  --num_points 2048 \
  --sampling_method poisson_disk
```

The script will:

- automatically load the `microsoft/TRELLIS-image-large` model (from Hugging Face);
- iterate over rendered images in `ShapeNetViPC-View` to generate meshes;
- sample meshes into point clouds and save them as `.pt` files,
  optionally writing visualizations under `.output`.

Once generation is done, `utils/dataloader.PCDataLoader` will automatically
look for generated point clouds under:

```text
ShapeNetViPC-Gen/trellis/<sampling_method>/num_points_<gen_points>/
```

## Training

The main training entry is `train.py`, and we recommend using `train.sh`
for single- or multi-GPU training on a single node.

### Single-GPU Training Example

```bash
bash train.sh \
  --config configs/ShapeNet-ViPC/PGNet/airplane.yaml \
  --gpu_num 1 \
  --gpu_ids 0
```

### Multi-GPU Training Example (single node)

```bash
bash train.sh \
  --config configs/ShapeNet-ViPC/PGNet/airplane.yaml \
  --gpu_num 4 \
  --gpu_ids 0,1,2,3
```

`train.sh` will set `CUDA_VISIBLE_DEVICES` for you, and for multi-GPU
it will call:

- `torchrun --standalone --nnodes=1 --nproc_per_node=<GPU_NUM> train.py`

Important training-related configs are all in the YAML file, e.g.:

- `training.global_batch_size`: logical global batch size;
- `training.gradient_accumulation_steps`: gradient accumulation steps;
- `training.max_steps` / `training.eval_steps`: max training steps & eval interval;
- `output.base_path`: root directory for all experiment outputs (default `output`).

#### GPU memory and gradient accumulation

- If you run into OOM (out of memory), increase `training.gradient_accumulation_steps` first.
- Keep `training.global_batch_size` unchanged to preserve the same optimization dynamics; changing it alters the effective batch size and can lead to different training results.
- The per-GPU physical batch size is computed as `global_batch_size // (gradient_accumulation_steps * WORLD_SIZE)` (see `train.py`). Here, `WORLD_SIZE` is the total number of participating GPUs/processes; in our single-node `train.sh` examples, `WORLD_SIZE == --gpu_num`.
- Ensure `global_batch_size % (gradient_accumulation_steps * WORLD_SIZE) == 0`.

After training, the default output structure looks like:

```text
output/
  PGNet_plane_poisson_disk_2048_YYYYMMDD_HHMMSS/
    ├── checkpoints/   # latest_step_*.pth, best_step_*.pth
    ├── configs/       # a snapshot of the config used for this run
    └── logs/          # TensorBoard logs
```


## Pretrained Checkpoints

We provide pretrained PGNet checkpoints on Hugging Face:

PGNet checkpoints dataset: [`Wang131/PGNet_ckpt`](https://huggingface.co/datasets/Wang131/PGNet_ckpt)

## Evaluation / Inference

`inference.py` is used for category-level evaluation on the full test set,
and reports three metrics:

- L2 Chamfer Distance (using the fine output);
- F-Score (threshold = 0.001);
- Earth Mover’s Distance (EMD).

Example usage:

```bash
python inference.py \
  -C configs/ShapeNet-ViPC/PGNet/airplane.yaml \
  -M /path/to/checkpoints/best_step_xxxx_xxx.pth \
  --device cuda:0
```

The script will iterate over all samples listed in `test_list.txt`,
compute per-sample metrics, and print the averaged results.

## Citation

If you find this repository or our paper helpful for your research,
please consider citing us:

```bibtex
@article{luo2025rethinking,
  title={Rethinking Multimodal Point Cloud Completion: A Completion-by-Correction Perspective},
  author={Luo, Wang and Wu, Di and Na, Hengyuan and Zhu, Yinlin and Hu, Miao and Quan, Guocong},
  journal={arXiv preprint arXiv:2511.12170},
  year={2025}
}
```

## Acknowledgements

This project primarily uses code from the following repositories:

- ViPC (View-Guided Point Cloud Completion): https://github.com/Hydrogenion/ViPC
- PoinTr (Diverse Point Cloud Completion with Geometry-Aware Transformers): https://github.com/yuxumin/PoinTr
- Microsoft TRELLIS (microsoft/TRELLIS-image-large): https://github.com/microsoft/TRELLIS
- FSC (FSC: Few-point Shape Completion): https://github.com/xianzuwu/FSC

We also acknowledge:

- PointNet++ and related CUDA extensions;

If you encounter issues or find bugs, feel free to open an Issue or submit a Pull Request.
