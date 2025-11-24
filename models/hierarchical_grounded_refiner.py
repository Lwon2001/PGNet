# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 SYSU/Wang Luo

import torch
import torch.nn as nn
from torch import einsum
from utils.modules import MLP_Res, knn_point
from pointnet2_ops.pointnet2_utils import grouping_operation
from utils.modules import MLP_Res, PointShuffle, MLP_CONV, coordinate_based_interpolation, feature_based_interpolation

class GroundedRefinementBlock(nn.Module):
    """
    Grounded Refinement Block: A block that refines point cloud features and coordinates
    by grounding them with features from a generative model and a partial input.
    """

    def __init__(
        self,
        feat_dim: int = 256,
        up_factor: int = 2,
        grb_idx: int = 0,
        k_interp: int = 8,
    ) -> None:
        super().__init__()
        self.grb_idx = grb_idx
        self.up_factor = up_factor
        self.k_interp = k_interp
        self.feat_dim = feat_dim
        self.q_feat_raw_dim = 3 + 2 * self.feat_dim

        # 1. Deep Residual Bottleneck Projector for q_raw
        self.q_feat_projector = nn.Sequential(
            MLP_Res(in_dim=self.q_feat_raw_dim, hidden_dim=512, out_dim=512),
            MLP_Res(in_dim=512, hidden_dim=self.feat_dim, out_dim=self.feat_dim)
        )

        # 2. Projector for features from the previous stage (K_prev)
        self.prev_feat_projector = MLP_Res(
            in_dim=self.feat_dim * 2,
            hidden_dim=self.feat_dim,
            out_dim=self.feat_dim,
        )
        
        # 3. Geometric processing module
        self.skip_transformer = LocalCrossAttention(
            in_channel=self.feat_dim,
            dim=self.feat_dim // 2,
            n_knn=16,
        )

        # 4. Coordinate prediction head
        self.mlp_feat_for_delta = MLP_CONV(
            in_channel=self.feat_dim * 2,
            layer_dims=[self.feat_dim, self.feat_dim * self.up_factor]
        )
        self.mlp_delta_coords = MLP_CONV(
            in_channel=self.feat_dim * self.up_factor,
            layer_dims=[max(32, self.feat_dim * self.up_factor // 2), 3 * self.up_factor]
        )
        self.delta_point_shuffle = PointShuffle(r=self.up_factor)

        # 5. Upsampler for coordinates
        self.up_sampler = nn.Upsample(scale_factor=up_factor)

    def construct_query_features_grb0(self, xyz, par_xyz, par_feat, gen_xyz, gen_feat):
        coord_feat = xyz
        par_interp = coordinate_based_interpolation(xyz, par_xyz, par_feat, k=self.k_interp)
        gen_interp = feature_based_interpolation(par_interp, par_feat, gen_feat, k=self.k_interp)
        return torch.cat([coord_feat, par_interp, gen_interp], dim=1)

    def construct_query_features_grb1(self, xyz, prev_xyz, prev_q_raw):
        start, end = 3, 3 + 2 * self.feat_dim
        prev_local = prev_q_raw[:, start:end, :]
        interp_local = coordinate_based_interpolation(xyz, prev_xyz, prev_local, k=self.k_interp)
        par_like = interp_local[:, : self.feat_dim, :]
        gen_like = interp_local[:, self.feat_dim :, :]
        return torch.cat([xyz, par_like, gen_like], dim=1)

    def forward(
        self,
        xyz: torch.Tensor,
        *,
        par_xyz=None,
        par_feat=None,
        gen_xyz=None,
        gen_feat=None,
        prev_xyz=None,
        prev_q_feat_raw=None,
        K_prev=None,
    ):
        if self.grb_idx == 0:
            q_raw = self.construct_query_features_grb0(
                xyz, par_xyz, par_feat, gen_xyz, gen_feat
            )
        else:
            assert prev_q_feat_raw is not None
            q_raw = self.construct_query_features_grb1(xyz, prev_xyz, prev_q_feat_raw)
        
        f1 = self.q_feat_projector(q_raw)
        query_for_geom_st = f1
        
        if self.grb_idx == 0:
            H_final = self.skip_transformer(
                query_pos=xyz,
                query_feat=query_for_geom_st,
                key_pos=xyz,
                key_feat=query_for_geom_st
            )
        else:
            assert prev_xyz is not None and K_prev is not None
            processed_K_prev = self.prev_feat_projector(K_prev)
            H_geometric = self.skip_transformer(
                query_pos=xyz,
                query_feat=query_for_geom_st,
                key_pos=prev_xyz,
                key_feat=processed_K_prev
            )
            H_final = H_geometric

        f2 = torch.cat([f1, H_final], dim=1)
    
        feat_delta_raw = self.mlp_feat_for_delta(f2)
        delta_features_presuffle = self.mlp_delta_coords(torch.relu(feat_delta_raw))   
        delta = self.delta_point_shuffle(delta_features_presuffle)

        xyz_up = self.up_sampler(xyz) + delta
        
        return xyz_up, f2, q_raw 

class LocalCrossAttention(nn.Module):
    """
    Cross-Attention Transformer to compute relation between two sets of points.
    """
    def __init__(self, in_channel, dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(LocalCrossAttention, self).__init__()
        self.n_knn = n_knn
        self.conv_query = nn.Conv1d(in_channel, dim, 1)
        self.conv_key = nn.Conv1d(in_channel, dim, 1)
        self.conv_value = nn.Conv1d(in_channel, dim, 1)
        
        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1)
        )

        self.conv_end = nn.Conv1d(dim, in_channel, 1)

    def forward(self, query_pos, query_feat, key_pos, key_feat, include_self=True):
        """
        Cross-attention forward pass.
        Args:
            query_pos: (B, 3, N_q) - coordinates of query points
            query_feat: (B, in_channel, N_q) - features of query points
            key_pos: (B, 3, N_k) - coordinates of key/value points
            key_feat: (B, in_channel, N_k) - features of key/value points
            include_self: boolean

        Returns:
            Tensor: (B, in_channel, N_q), shape context feature
        """
        identity = query_feat

        query = self.conv_query(query_feat)  # (B, dim, N)
        key = self.conv_key(key_feat)  # (B, dim, N_prev)
        value = self.conv_value(key_feat)  # (B, dim, N_prev)
        b, dim, n_q = query.shape

        query_pos_flipped = query_pos.permute(0, 2, 1).contiguous()  # (B, N, 3)    
        key_pos_flipped = key_pos.permute(0, 2, 1).contiguous()  # (B, N_prev, 3)
        if not include_self:
            idx_knn = knn_point(self.n_knn + 1, key_pos_flipped, query_pos_flipped).int() # (B, N, n_knn)
            idx_knn = idx_knn[:, :, 1:].contiguous()
        else:
            idx_knn = knn_point(self.n_knn, key_pos_flipped, query_pos_flipped).int() # (B, N, n_knn)

        key_grouped = grouping_operation(key, idx_knn)  # b, dim, n_q, n_knn
        pos_grouped = grouping_operation(key_pos, idx_knn)  # b, 3, n_q, n_knn

        pos_rel = query_pos.reshape((b, -1, n_q, 1)) - pos_grouped
        pos_embedding = self.pos_mlp(pos_rel)

        qk_rel = query.reshape((b, -1, n_q, 1)) - key_grouped
        
        attention = self.attn_mlp(qk_rel + pos_embedding)
        attention = torch.softmax(attention, -1)

        value_grouped = grouping_operation(value, idx_knn)
        value_with_pos = value_grouped + pos_embedding

        agg = einsum('b c i j, b c i j -> b c i', attention, value_with_pos)
        y = self.conv_end(agg)

        return y + identity 

class HierarchicalGroundedRefiner(nn.Module):
    def __init__(self, feat_dim, num_coarse, config, num_fine_pc_target):
        """
        Hierarchical Grounded Refiner: A hierarchical decoder that progressively
        upsamples a coarse point cloud by grounding it with features from a generative
        model and a partial input.
        Args:
            feat_dim: Dimension of features used throughout the refiner.
            num_coarse: Number of points in the initial coarse point cloud.
            config: A dictionary containing decoder parameters like 'up_factors' and 'k_interp'.
            num_fine_pc_target: The target number of fine points for verification.
        """
        super(HierarchicalGroundedRefiner, self).__init__()
        
        self.num_coarse = num_coarse
        self.up_factors = config['up_factors']
        self.k_interp = config['k_interp']
        
        self.num_final = num_coarse
        for factor in self.up_factors:
            self.num_final *= factor
            
        if self.num_final != num_fine_pc_target:
            print(f"[HGR WARNING] Final points ({self.num_final}) mismatch target ({num_fine_pc_target}).")
        
        print(f"[HierarchicalGroundedRefiner] Coarse: {num_coarse}, "
              f"UpFactors: {self.up_factors}, Final: {self.num_final}")
        
        self.grb_layers = nn.ModuleList()
        for i, factor in enumerate(self.up_factors):
            grb = GroundedRefinementBlock(
                feat_dim=feat_dim,
                up_factor=factor,
                grb_idx=i,
                k_interp=self.k_interp
            )
            self.grb_layers.append(grb)

    def forward(self, coarse_xyz, par_xyz, par_feat, gen_xyz, gen_feat):
        """
        Args:
            coarse_xyz: (B, N_coarse, 3) coarse point cloud
            par_xyz: (B, N_par, 3) partial point cloud proxy coordinates
            par_feat: (B, N_par, C_local) partial point cloud local features
            gen_xyz: (B, N_gen, 3) generated point cloud proxy coordinates
            gen_feat: (B, N_gen, C_local) generated point cloud local features
        Returns:
            all_pcds: List[Tensor] point clouds at all resolutions
        """
        B = coarse_xyz.shape[0]

        # Transpose all inputs to (B, C, N) format for internal processing
        coarse_xyz_t = coarse_xyz.transpose(1, 2).contiguous()
        par_xyz_t = par_xyz.transpose(1, 2).contiguous()
        par_feat_t = par_feat.transpose(1, 2).contiguous()
        gen_xyz_t = gen_xyz.transpose(1, 2).contiguous()
        gen_feat_t = gen_feat.transpose(1, 2).contiguous()
        
        current_xyz = coarse_xyz_t
        all_pcds = [coarse_xyz]
        
        prev_xyz = None
        prev_q_feat = None
        K_prev = None
        
        for i, grb_layer in enumerate(self.grb_layers):
            if i == 0:
                refined_xyz, K_curr, q_feat = grb_layer(
                    xyz=current_xyz,
                    par_xyz=par_xyz_t,
                    par_feat=par_feat_t,
                    gen_xyz=gen_xyz_t,
                    gen_feat=gen_feat_t,
                    K_prev=None
                )
            else:
                refined_xyz, K_curr, q_feat = grb_layer(
                    xyz=current_xyz,
                    prev_xyz=prev_xyz,
                    prev_q_feat_raw=prev_q_feat,
                    K_prev=K_prev
                )
            
            prev_xyz = current_xyz
            prev_q_feat = q_feat
            current_xyz = refined_xyz
            K_prev = K_curr
            
            current_pcd = current_xyz.transpose(1, 2).contiguous()
            all_pcds.append(current_pcd)
        
        return all_pcds 