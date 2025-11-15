# PointNet++ Set Abstraction 与 Feature Propagation 实现（附中文注释）
# 代码基于原作者提供的官方实现，结合论文《PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space》进行逐行注释。
# -----------------------------------------------------------------------------
# Copyright (c) <original license>. This file adds explanatory comments in Chinese.
# -----------------------------------------------------------------------------

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils

# =============================================================================
# 1. 构建共享的 MLP (mini‑PointNet)
#    论文 §3.2 PointNet layer: 对局部区域中的每个点独立地学习高维表示，
#    再通过 max‑pool 汇聚；这里的 1×1 卷积等价于对每个点的全连接层，
#    且权重在所有点之间共享 (Shared MLP)。
# =============================================================================

def build_shared_mlp(mlp_spec: List[int], bn: bool = True) -> nn.Sequential:
    """根据给定的通道规格构造 Shared MLP。

    Args:
        mlp_spec (List[int]): 输入/输出通道序列，例如 [in_c, 64, 128, 256]。
        bn (bool): 是否在 Conv2d 之后使用 BatchNorm2d。

    Returns:
        nn.Sequential: 按 Conv2d -> BN -> ReLU 顺序堆叠的模块。
    """
    layers: List[nn.Module] = []
    for i in range(1, len(mlp_spec)):
        # 1×1 卷积，相当于逐点线性变换；输入形状 (B, Cin, npoint, nsample)
        layers.append(nn.Conv2d(mlp_spec[i - 1], mlp_spec[i], kernel_size=1, bias=not bn))
        if bn:
            layers.append(nn.BatchNorm2d(mlp_spec[i]))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

# =============================================================================
# 2. Set Abstraction 基类 _PointnetSAModuleBase
#    负责 (1) FPS 采样得到 new_xyz, (2) 调用 QueryAndGroup 进行邻域分组，
#    (3) 对每个 scale 应用 mini‑PointNet 并做池化，最后拼接各尺度特征。
# =============================================================================

class _PointnetSAModuleBase(nn.Module):
    def __init__(self):
        super().__init__()
        # 采样点个数 (None 表示不采样，直接 GroupAll)
        self.npoint: Optional[int] = None
        # 不同半径/采样数对应的 QueryAndGroup 模块列表
        self.groupers: Optional[nn.ModuleList] = None
        # 对应每个分组尺度的 mini‑PointNet (Shared MLP)
        self.mlps: Optional[nn.ModuleList] = None

    def forward(self, xyz: torch.Tensor, features: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播。

        Args:
            xyz (Tensor): 形状 (B, N, 3)，输入点坐标。
            features (Tensor|None): 形状 (B, C, N)，输入点特征；第一次 SA 层时为 None。

        Returns:
            new_xyz (Tensor|None): 采样后的中心点 (B, npoint, 3)。若 npoint=None 则返回 None。
            new_features (Tensor): 每个中心点的特征，维度 (B, Σ mlp[-1], npoint)。
        """
        # -------------------------------------------------
        # 1) FPS (Farthest Point Sampling) 选取 npoint 个中心点
        #    论文 §3.2 Sampling layer，保证覆盖性更好。
        # -------------------------------------------------
        xyz_flipped = xyz.transpose(1, 2).contiguous()  # (B, 3, N) —— pointnet2_ops 要求通道在前
        if self.npoint is not None:
            # indices: (B, npoint)
            fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
            # gather_operation: 根据 fps_idx 从 xyz_flipped 中取点
            new_xyz = pointnet2_utils.gather_operation(xyz_flipped, fps_idx).transpose(1, 2).contiguous()
        else:
            new_xyz = None  # MSG 顶层可能不需要再采样

        # -------------------------------------------------
        # 2) 对每个尺度执行: 分组 -> Shared MLP -> max pooling
        # -------------------------------------------------
        new_features_list = []
        for i in range(len(self.groupers)):
            # QueryAndGroup: 返回 (B, C+3, npoint, nsample)
            #   若 features=None，则只返回 XYZ
            grouped_features = self.groupers[i](xyz, new_xyz, features)

            # 应用 Shared MLP: (B, mlp[-1], npoint, nsample)
            grouped_features = self.mlps[i](grouped_features)

            # 对 nsample 维度做 max‑pooling，得到局部区域 (Ball) 的全局特征
            grouped_features = F.max_pool2d(grouped_features, kernel_size=[1, grouped_features.size(3)])
            grouped_features = grouped_features.squeeze(-1)  # (B, mlp[-1], npoint)
            new_features_list.append(grouped_features)

        # 按通道拼接多尺度特征
        new_features = torch.cat(new_features_list, dim=1)
        return new_xyz, new_features

# =============================================================================
# 3. PointnetSAModuleMSG (Multi‑Scale Grouping)
#    对应论文 §3.3 Multi‑scale grouping (MSG)，在同一 SA Level 内部
#    使用不同半径 / 采样数，以增强对非均匀稀疏点云的鲁棒性。
# =============================================================================

class PointnetSAModuleMSG(_PointnetSAModuleBase):
    def __init__(self, npoint: int, radii: List[float], nsamples: List[int], mlps: List[List[int]], bn: bool = True, use_xyz: bool = True):
        super().__init__()
        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for radius, nsample, mlp_spec in zip(radii, nsamples, mlps):
            # 3.1) 构造分组器
            if npoint is not None:
                self.groupers.append(pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz))
            else:
                # 顶层: 将全部点视为一个 group
                self.groupers.append(pointnet2_utils.GroupAll(use_xyz))

            # 3.2) 如果使用坐标，则把 3 维 xyz 拼在特征前
            if use_xyz:
                mlp_spec = mlp_spec.copy()  # 避免原列表被后续修改
                mlp_spec[0] += 3
            self.mlps.append(build_shared_mlp(mlp_spec, bn))

# =============================================================================
# 4. PointnetSAModule (Single‑Scale Grouping)
#    为方便起见，将单尺度参数包装成 MSG 的特殊情况。
# =============================================================================

class PointnetSAModule(PointnetSAModuleMSG):
    def __init__(self, mlp: List[int], npoint: int = None, radius: float = None, nsample: int = None, bn: bool = True, use_xyz: bool = True):
        super().__init__(npoint=npoint, radii=[radius], nsamples=[nsample], mlps=[mlp], bn=bn, use_xyz=use_xyz)

# =============================================================================
# 5. Feature Propagation (FP) Module
#    对应论文 §3.4 Point Feature Propagation，在解码阶段把特征插值
#    回所有原始点，用于分割等密集预测任务。
# =============================================================================

class PointnetFPModule(nn.Module):
    def __init__(self, mlp: List[int], bn: bool = True):
        super().__init__()
        # 构建 (C_in -> ... -> C_out) Shared MLP，输入形状为 (B, C_in, N, 1)
        self.mlp = build_shared_mlp(mlp, bn=bn)

    def forward(self, unknown: torch.Tensor, known: torch.Tensor, unknow_feats: torch.Tensor, known_feats: torch.Tensor) -> torch.Tensor:
        """特征插值与融合。

        Args:
            unknown (Tensor): (B, n, 3)，待插值的点坐标 (上一级更密集)。
            known (Tensor):   (B, m, 3)，已知特征的点坐标 (本级采样中心)。
            unknow_feats (Tensor|None): (B, C1, n)，来自 skip connection 的特征。
            known_feats (Tensor): (B, C2, m)，来自上一层解码的特征。

        Returns:
            Tensor: (B, mlp[-1], n)，输出更新后的 unknown 点特征。
        """
        # -------------------------------------------------
        # 1) 三邻域插值 (three_nn + three_interpolate)
        #    论文 Eq.(2) 逆距离加权插值 (k=3)。若 known==None 则直接广播。
        # -------------------------------------------------
        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)        # (B, n, 3)
            dist_recip = 1.0 / (dist + 1e-8)                           # 避免除零
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm                                 # 权重归一化
            interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)  # (B, C2, n)
        else:
            # 顶层特征直接扩展
            interpolated_feats = known_feats.expand(*known_feats.size()[:2], unknown.size(1))

        # -------------------------------------------------
        # 2) Skip 连接: 将插值结果与 encoder 同层特征拼接
        # -------------------------------------------------
        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)  # (B, C2+C1, n)
        else:
            new_features = interpolated_feats

        # -------------------------------------------------
        # 3) Shared MLP (1×1 Conv) -> 更新特征
        # -------------------------------------------------
        new_features = new_features.unsqueeze(-1)            # (B, C, n, 1)
        new_features = self.mlp(new_features)                # (B, C_out, n, 1)
        return new_features.squeeze(-1)                      # (B, C_out, n)
