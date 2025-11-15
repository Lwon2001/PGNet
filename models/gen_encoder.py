import torch
import torch.nn as nn
from pytorch3d.ops import knn_points

from timm.layers.drop import DropPath
from timm.layers.weight_init import trunc_normal_

from utils.modules import MLP_Res, RelativePos, Mlp, Attention, CrossAttention
from models.dgcnn_group import DGCNN_Grouper

import numpy as np


class GroundingTransformer(nn.Module):
    """
    A transformer block that performs gated grounding. It adaptively fuses
    self-attention (internal reasoning) with cross-attention (reality context)
    via a learned gating mechanism.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1_self = norm_layer(dim)
        self.norm1_cross_q = norm_layer(dim)
        self.norm1_cross_v = norm_layer(dim)
        
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            attn_drop=0., proj_drop=drop
        )
        self.cross_attn = CrossAttention(
            dim=dim, out_dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=0., proj_drop=drop
        )
        
        # Gating network
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

    def forward(self, x, reality_feat):
        # x: generative prior features
        # reality_feat: features from the partial input (ground truth)
        
        # 1. Internal Reasoning (Self-Attention)
        norm_x_self = self.norm1_self(x)
        self_attn_out = self.attn(norm_x_self)
        
        # 2. Reality Context (Cross-Attention)
        norm_x_cross = self.norm1_cross_q(x)
        norm_reality = self.norm1_cross_v(reality_feat)
        cross_attn_out = self.cross_attn(norm_x_cross, norm_reality)
        
        # 3. Gated Fusion
        gate_input = torch.cat([self_attn_out, cross_attn_out], dim=-1)
        gate = self.gate_mlp(gate_input)
        
        # 4. Adaptive Fusion
        fused_feat = (1 - gate) * self_attn_out + gate * cross_attn_out
        
        x = x + self.drop_path(fused_feat)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class GenEncoder(nn.Module):
    def __init__(self, config, output_dim, in_chans=3):
        super().__init__()
        self.num_features = self.embed_dim = config['embed_dim']
        
        print('Grounding Transformer armed.')

        self.grouper = DGCNN_Grouper(k_values=config['dgcnn_k_values'])

        pos_embed_type = config.get('pos_embed', 'relative')
        if pos_embed_type == 'relative':
            self.pos_embed = RelativePos(out_ch=self.embed_dim)
        elif pos_embed_type == 'absolute':
            self.pos_embed = nn.Sequential(
                nn.Conv1d(in_chans, 128, 1),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv1d(128, self.embed_dim, 1)
            )
        else:
            self.pos_embed = None

        self.input_proj = nn.Sequential(
            nn.Conv1d(128, self.embed_dim, 1),
            nn.BatchNorm1d(self.embed_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.embed_dim, self.embed_dim, 1)
        )

        self.transformer = GroundingTransformer(
            dim=self.embed_dim,
            num_heads=config['num_heads'],
            mlp_ratio=config['mlp_ratio'],
            qkv_bias=config.get('qkv_bias', False),
            qk_scale=config.get('qk_scale'),
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

    def forward(self, inpc, reality_guided_feature):
        bs = inpc.size(0)
        coor, f = self.grouper(inpc.transpose(1,2).contiguous())
        x = self.input_proj(f).transpose(1,2)
        if self.pos_embed is not None:
            pos = self.pos_embed(coor).transpose(1,2)
            x = x + pos

        x = self.transformer(x, reality_guided_feature)
                
        pre_local_feat = x
        
        local_feature_t = self.output_proj(pre_local_feat.transpose(1,2))
        global_feature = torch.max(local_feature_t, dim=-1)[0]
        local_feature = local_feature_t.transpose(1,2).contiguous()

        # Return gen_xyz which are the proxy coordinates from the grouper
        return local_feature, global_feature, coor.transpose(1,2).contiguous() 