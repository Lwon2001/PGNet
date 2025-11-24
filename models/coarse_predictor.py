# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 SYSU/Wang Luo

import torch
import torch.nn as nn
from utils.modules import MLP_Res, PointShuffle, cross_attention

class GroundedCoarsePredictor(nn.Module):
    def __init__(self, feat_dim, num_pc, config, coarse_feat_dim=128):
        super(GroundedCoarsePredictor, self).__init__()
        self.num_pc = num_pc
        self.coarse_feat_dim = coarse_feat_dim
        
        # Global feature fusion
        cross_attention_config = config
        self.global_feat_fuser = cross_attention(
            d_model=feat_dim,
            d_model_out=feat_dim,
            nhead=cross_attention_config['nhead'],
            dim_feedforward=feat_dim,
            dropout=cross_attention_config['dropout']
        )
        
        self.unfolding_mlp = nn.Sequential(
            nn.Conv1d(feat_dim, feat_dim * 2, 1),
            nn.ReLU(inplace=True),
            MLP_Res(in_dim=feat_dim * 2, hidden_dim=feat_dim * 2, out_dim=feat_dim * 2),
            nn.Conv1d(feat_dim * 2, self.coarse_feat_dim * self.num_pc, 1)
        )
        self.point_shuffle = PointShuffle(r=self.num_pc)
        
        if config is None:
            raise ValueError("config is required for GroundedCoarsePredictor")
            
        self.par_feat_projector = nn.Conv1d(feat_dim, self.coarse_feat_dim, 1)
        self.grounding_cross_attention = cross_attention(
            d_model=self.coarse_feat_dim,
            d_model_out=self.coarse_feat_dim,
            nhead=config['nhead'],
            dim_feedforward=256,
            dropout=config['dropout']
        )
        self.fusion_mlp = MLP_Res(in_dim=self.coarse_feat_dim * 2 + feat_dim, hidden_dim=512, out_dim=self.coarse_feat_dim)
        self.coord_predictor = nn.Sequential(
            nn.Conv1d(self.coarse_feat_dim, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, par_global, gen_global, par_feat_transposed):
        # 1. Fuse global features.
        par_global_expanded = par_global.unsqueeze(-1)
        gen_global_expanded = gen_global.unsqueeze(-1)
        fused_global = self.global_feat_fuser(
            par_global_expanded, gen_global_expanded
        )
        
        unfolded_feat = self.unfolding_mlp(fused_global)
        proposed_seeds = self.point_shuffle(unfolded_feat)
        par_feat_proj = self.par_feat_projector(par_feat_transposed)
        grounding_context = self.grounding_cross_attention(proposed_seeds, par_feat_proj)
        global_feat_expanded = fused_global.expand(-1, -1, self.num_pc)
        fused_feat = torch.cat([global_feat_expanded, proposed_seeds, grounding_context], dim=1)
        final_feat = self.fusion_mlp(fused_feat)
        coarse_pcd = self.coord_predictor(final_feat)
        return coarse_pcd