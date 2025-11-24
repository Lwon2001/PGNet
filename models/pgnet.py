# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 SYSU/Wang Luo

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gen_encoder import GenEncoder
from models.partial_encoder import PartialEncoder
from models.hierarchical_grounded_refiner import HierarchicalGroundedRefiner
from models.coarse_predictor import GroundedCoarsePredictor
from utils.modules import MLP_Res, PointShuffle, cross_attention

class PGNet(nn.Module):
    def __init__(self, config):
        super(PGNet, self).__init__()

        model_config = config['model']
        dataset_config = config['dataset']

        self.num_coarse_pc = model_config['num_coarse_pc']
        self.feat_dim = model_config['feat_dim']
        self.num_fine_pc_target = dataset_config['fine_points']

        # Encoder for generative prior
        self.gen_encoder = GenEncoder(
            config=model_config['gen_encoder'],
            output_dim=self.feat_dim
        )

        # Encoder for partial cloud
        self.par_encoder = PartialEncoder(
            config=model_config['partial_encoder'],
            output_dim=self.feat_dim
        )

        # Global feature fusion
        cross_attention_config = model_config['cross_attention']
        self.global_feat_fuser = cross_attention(
            d_model=self.feat_dim,
            d_model_out=self.feat_dim,
            nhead=cross_attention_config['nhead'],
            dim_feedforward=self.feat_dim,
            dropout=cross_attention_config['dropout']
        )

        # Coarse point cloud prediction
        self.coarse_predictor = GroundedCoarsePredictor(
            feat_dim=self.feat_dim,
            num_pc=self.num_coarse_pc,
            config=cross_attention_config
        )

        # Decoder for final reconstruction
        self.decoder = HierarchicalGroundedRefiner(
            feat_dim=self.feat_dim,
            num_coarse=self.num_coarse_pc,
            config=model_config['decoder'],
            num_fine_pc_target=self.num_fine_pc_target
        )

    def forward(self, gen_pcds, par_pcds):
        # 1. Encode partial and generated point clouds.
        par_local, par_global, par_xyz, par_pre_local_feat = self.par_encoder(par_pcds)
        gen_local, gen_global, gen_xyz = self.gen_encoder(gen_pcds, reality_guided_feature=par_pre_local_feat)

        # 2. Predict coarse point cloud, fusing global features internally.
        coarse_pcd_transposed = self.coarse_predictor(
            par_global=par_global,
            gen_global=gen_global,
            par_feat_transposed=par_local.transpose(1, 2).contiguous()
        )
        coarse_pcd = coarse_pcd_transposed.transpose(1, 2).contiguous()

        # 3. Decode the fine point cloud, handling transpositions internally.
        all_pcds = self.decoder(
            coarse_xyz=coarse_pcd,
            par_xyz=par_xyz,
            par_feat=par_local,
            gen_xyz=gen_xyz,
            gen_feat=gen_local
        )

        # The last element is the fine pcd
        # fine_pcd = all_pcds[-1]

        # Return a dictionary for clarity
        output = {
            "all_pcds": all_pcds, # list of pcds from coarse to fine
            "par_xyz_downsampled": par_xyz,
            "gen_xyz_downsampled": gen_xyz,
        }
        return output 