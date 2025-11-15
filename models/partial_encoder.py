import torch
import torch.nn as nn
from pytorch3d.ops import knn_points

from timm.layers.drop import DropPath
from timm.layers.weight_init import trunc_normal_

from utils.modules import MLP_Res, RelativePos, knn_point, Mlp, Attention
from models.dgcnn_group import DGCNN_Grouper

import numpy as np

def get_knn_index(coor_q, coor_k=None):
    coor_k = coor_k if coor_k is not None else coor_q
    batch_size, _, num_points = coor_q.size()
    num_points_k = coor_k.size(2)

    with torch.no_grad():
        idx = knn_point(8, coor_k.transpose(-1, -2).contiguous(), coor_q.transpose(-1, -2).contiguous()) # B G M
        idx = idx.transpose(-1, -2).contiguous()
        idx_base = torch.arange(0, batch_size, device=coor_q.device).view(-1, 1, 1) * num_points_k
        idx = idx + idx_base
        idx = idx.view(-1)
    
    return idx  # bs*k*np

def get_graph_feature(x, knn_index, x_q=None):
        k = 8
        batch_size, num_points, num_dims = x.size()
        num_query = x_q.size(1) if x_q is not None else num_points
        feature = x.view(batch_size * num_points, num_dims)[knn_index, :]
        feature = feature.view(batch_size, k, num_query, num_dims)
        x = x_q if x_q is not None else x
        x = x.view(batch_size, 1, num_query, num_dims).expand(-1, k, -1, -1)
        feature = torch.cat((feature - x, x), dim=-1)
        return feature  # b k np c


class SalientTransformer(nn.Module):
    """
    An adaptive transformer block that fuses global context (from self-attention)
    and local geometry (from graph features) via a learned salience gate.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=0., proj_drop=drop)
        
        # KNN/Graph-based stream
        self.knn_map = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # Gating mechanism
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, knn_index):
        norm_x = self.norm1(x)
        
        # 1. Context Stream (Self-Attention)
        attn_out = self.attn(norm_x)

        # 2. Geometry Stream (Graph Features)
        knn_f = get_graph_feature(norm_x, knn_index)
        knn_out = self.knn_map(knn_f).max(dim=1, keepdim=False)[0]

        # 3. Salience Gating
        gate_input = torch.cat([attn_out, knn_out], dim=-1)
        gate = self.gate_mlp(gate_input)

        # 4. Adaptive Fusion
        fused_feat = (1 - gate) * attn_out + gate * knn_out
        
        x = x + self.drop_path(fused_feat)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PartialEncoder(nn.Module):
    """
    A transformer that uses SalientFeatureBlocks to create a "High-Fidelity
    Calibration Blueprint" from a partial point cloud, adaptively focusing on
    geometrically significant regions.
    """
    def __init__(self, config, output_dim, in_chans=3):
        super().__init__()
        self.num_features = self.embed_dim = config['embed_dim']
        
        self.grouper = DGCNN_Grouper(k_values=config['dgcnn_k_values'])

        pos_embed_type = config.get('pos_embed', 'relative')
        if pos_embed_type == 'relative':
            self.pos_embed = RelativePos(out_ch=self.embed_dim)
        else:
            self.pos_embed = None

        self.input_proj = nn.Sequential(
            nn.Conv1d(128, self.embed_dim, 1),
            nn.BatchNorm1d(self.embed_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.embed_dim, self.embed_dim, 1)
        )

        self.transformer = SalientTransformer(
            dim=self.embed_dim,
            num_heads=config['num_heads'],
            mlp_ratio=config['mlp_ratio'],
            qkv_bias=config.get('qkv_bias', False),
            qk_scale=config.get('qk_scale', 0.),
            drop=config['drop_rate'])

        self.output_proj = nn.Sequential(
            MLP_Res(in_dim=self.embed_dim, hidden_dim=512, out_dim=512),
            MLP_Res(in_dim=512, hidden_dim=output_dim, out_dim=output_dim)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight.data, gain=1)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, inpc):
        coor, f = self.grouper(inpc.transpose(1,2).contiguous())
        knn_index = get_knn_index(coor)
        x = self.input_proj(f).transpose(1,2)

        if self.pos_embed is not None:
            pos = self.pos_embed(coor).transpose(1,2)
            x = x + pos

        x = self.transformer(x, knn_index)
        
        pre_local_feat = x
        
        local_feature_t = self.output_proj(pre_local_feat.transpose(1,2))
        global_feature = torch.max(local_feature_t, dim=-1)[0]
        local_feature = local_feature_t.transpose(1,2).contiguous()

        return local_feature, global_feature, coor.transpose(1,2).contiguous(), pre_local_feat 