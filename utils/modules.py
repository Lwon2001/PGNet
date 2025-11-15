import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_points


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.k_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.v_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, v):
        B, N, _ = q.shape
        C = self.out_dim
        k = v
        NK = k.size(1)
        q = self.q_map(q).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_map(k).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_map(v).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP_Res(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=None, out_dim=128):
        super(MLP_Res, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.conv_1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv_2 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.conv_shortcut = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (B, out_dim, n)
        """
        shortcut = self.conv_shortcut(x)
        out = self.conv_2(torch.relu(self.conv_1(x))) + shortcut
        return out

class MLP_CONV(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP_CONV, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)
    
class PointShuffle(nn.Module):
    """(B, C × r, N)  ➜  (B, C, N × r)"""
    def __init__(self, r: int):
        super().__init__()
        self.r = r
    def forward(self, x: torch.Tensor):
        B, Cr, N = x.shape
        assert Cr % self.r == 0, "channel not divisible by r"
        C = Cr // self.r
        return (x.view(B, self.r, C, N)
                 .permute(0, 2, 3, 1)
                 .reshape(B, C, N * self.r))
        



def coordinate_based_interpolation(query_xyz, support_xyz, support_feat, k=8):
    """
    基于坐标距离的特征插值 (使用PyTorch3D高效KNN)
    Args:
        query_xyz: (B, 3, N_query) 查询点坐标
        support_xyz: (B, 3, N_support) 支撑点坐标  
        support_feat: (B, C, N_support) 支撑点特征
        k: 近邻数量
    Returns:
        interpolated_feat: (B, C, N_query) 插值后的特征
    """
    B, _, N_query = query_xyz.shape
    _, C, N_support = support_feat.shape
    
    # 转换为PyTorch3D期望的格式 (B, N, 3)
    query_xyz_t = query_xyz.transpose(1, 2).contiguous()  # (B, N_query, 3)
    support_xyz_t = support_xyz.transpose(1, 2).contiguous()  # (B, N_support, 3)
    
    # 使用PyTorch3D的高效KNN实现
    knn_result = knn_points(query_xyz_t, support_xyz_t, K=k, return_nn=False)
    knn_dists = knn_result.dists  # (B, N_query, k) - 已经是平方距离
    knn_idx = knn_result.idx      # (B, N_query, k) - 已经是int64类型
    
    # 计算逆距离权重
    # knn_dists是平方距离，我们取平方根得到欧氏距离
    dists = torch.sqrt(knn_dists + 1e-8)  # 添加小值避免除零
    weights = 1.0 / (dists + 1e-8)       # 逆距离权重
    weights = weights / weights.sum(dim=-1, keepdim=True)  # 归一化
    
    # 使用KNN索引收集特征
    # support_feat: (B, C, N_support) -> (B, N_support, C)
    support_feat_t = support_feat.transpose(1, 2).contiguous()  # (B, N_support, C)
    
    # 扩展索引以匹配特征维度
    knn_idx_expanded = knn_idx.unsqueeze(-1).expand(-1, -1, -1, C)  # (B, N_query, k, C)
    
    # 收集KNN特征
    knn_feat = torch.gather(
        support_feat_t.unsqueeze(1).expand(-1, N_query, -1, -1),  # (B, N_query, N_support, C)
        2,  # dim=2 对应N_support维度
        knn_idx_expanded
    )  # (B, N_query, k, C)
    
    # 加权聚合
    interpolated_feat = (knn_feat * weights.unsqueeze(-1)).sum(dim=2)  # (B, N_query, C)
    
    # 转换回原始格式
    return interpolated_feat.transpose(1, 2).contiguous()  # (B, C, N_query)


def feature_based_interpolation(query_feat, support_feat, support_target, k=8):
    """
    基于特征距离的特征插值
    Args:
        query_feat: (B, C, N_query) 查询特征
        support_feat: (B, C, N_support) 支撑特征
        support_target: (B, C_target, N_support) 要插值的目标特征
        k: 近邻数量
    Returns:
        interpolated_target: (B, C_target, N_query) 插值后的目标特征
    """
    B, C, N_query = query_feat.shape
    _, C_target, N_support = support_target.shape
    
    # L2归一化特征以获得更好的相似度计算
    query_feat_norm = F.normalize(query_feat, dim=1)      # (B, C, N_query)
    support_feat_norm = F.normalize(support_feat, dim=1)  # (B, C, N_support)
    
    # 计算特征相似度矩阵 (使用点积)
    # 转换为 (B, N, C) 格式便于矩阵乘法
    query_feat_t = query_feat_norm.transpose(1, 2)      # (B, N_query, C)
    support_feat_t = support_feat_norm.transpose(1, 2)  # (B, N_support, C)
    
    # 计算相似度 query @ support^T
    similarity = torch.bmm(query_feat_t, support_feat_t.transpose(1, 2))  # (B, N_query, N_support)
    
    # 找到k个最相似的特征
    knn_similarity, knn_idx = torch.topk(similarity, k, dim=-1)  # (B, N_query, k)
    
    # 使用softmax将相似度转换为权重
    weights = F.softmax(knn_similarity * 10, dim=-1)  # 温度为10的softmax
    
    # 使用KNN索引收集目标特征
    # support_target: (B, C_target, N_support) -> (B, N_support, C_target)
    support_target_t = support_target.transpose(1, 2).contiguous()  # (B, N_support, C_target)
    
    # 扩展索引以匹配目标特征维度
    knn_idx_expanded = knn_idx.unsqueeze(-1).expand(-1, -1, -1, C_target)  # (B, N_query, k, C_target)
    
    # 收集KNN目标特征
    knn_target = torch.gather(
        support_target_t.unsqueeze(1).expand(-1, N_query, -1, -1),  # (B, N_query, N_support, C_target)
        2,  # dim=2 对应N_support维度
        knn_idx_expanded
    )  # (B, N_query, k, C_target)
    
    # 加权聚合
    interpolated_target = (knn_target * weights.unsqueeze(-1)).sum(dim=2)  # (B, N_query, C_target)
    
    # 转换回原始格式
    return interpolated_target.transpose(1, 2).contiguous()  # (B, C_target, N_query) 

def knn_point(nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    knn_result = knn_points(new_xyz, xyz, K=nsample, return_nn=False)
    return knn_result.idx

class RelativePos(nn.Module):
    def __init__(self, in_ch=6, out_ch=768, k=16):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Conv2d(in_ch, 128, 1),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 128, 1),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, out_ch, 1)
        )
    def forward(self, xyz):              # xyz: B,3,N
        xyz_t = xyz.transpose(1,2).contiguous()  # B,N,3
        idx = knn_point(self.k, xyz_t, xyz_t)  # B,N,k
        batch_size, num_points, _ = xyz_t.shape
        batch_indices = torch.arange(batch_size, device=xyz.device).view(batch_size, 1, 1).expand(-1, num_points, self.k)
        neighbors = xyz_t[batch_indices, idx]  # B,N,k,3
        neighbors = neighbors.permute(0, 3, 1, 2).contiguous()
        center_points = xyz.unsqueeze(-1).expand(-1, -1, -1, self.k)  # B,3,N,k

        diff = neighbors - center_points  # B,3,N,k
        geometric_info = torch.cat((diff, center_points), dim=1) # B,6,N,k

        feat = self.mlp(geometric_info).max(-1)[0]  # B,out_ch,N
        return feat
    
class cross_attention(nn.Module):

    def __init__(self, d_model=256, d_model_out=256, nhead=4, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model_out, nhead, dropout=dropout)
        self.linear11 = nn.Linear(d_model_out, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model_out)

        self.norm12 = nn.LayerNorm(d_model_out)
        self.norm13 = nn.LayerNorm(d_model_out)

        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)

        self.activation1 = torch.nn.GELU()

        self.input_proj = nn.Conv1d(d_model, d_model_out, kernel_size=1)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src1, src2, pos=None):
        # pdb.set_trace()
        src1 = self.input_proj(src1)
        src2 = self.input_proj(src2)
        b, c, _ = src1.shape
        src1 = src1.reshape(b, c, -1).permute(2, 0, 1)
        src2 = src2.reshape(b, c, -1).permute(2, 0, 1)
        src1 = self.norm13(src1)
        src2 = self.norm13(src2)
        q  = self.with_pos_embed(src1, pos)
        src12 = self.multihead_attn(query=q,
                                     key=src2,
                                     value=src2)[0]  #  以 src1（加上位置编码）作为查询（query），src2 作为键（key）和值（value），计算注意力输出
        src1 = src1 + self.dropout12(src12)  # 将注意力结果经过 dropout 后与原始的 src1 相加，再经过 norm12 归一化。
        src1 = self.norm12(src1)
        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))   # 对上一步的输出应用前馈网络（全连接层、激活、dropout、第二个全连接层、dropout），再加上之前的输入形成残差连接
        src1 = src1 + self.dropout13(src12)
        src1 = src1.permute(1, 2, 0)  # 最后将张量重新排列回 [batch, channels, length] 的格式，作为最终输出

        return src1

class self_attention(nn.Module):

    def __init__(self, d_model=256, d_model_out=256, nhead=4, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model_out, nhead, dropout=dropout)
        self.linear11 = nn.Linear(d_model_out, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model_out)
        self.norm12 = nn.LayerNorm(d_model_out)
        self.norm13 = nn.LayerNorm(d_model_out)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)
        self.activation1 = torch.nn.GELU()
        self.input_proj = nn.Conv1d(d_model, d_model_out, kernel_size=1)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src1, pos=None):
        src1 = self.input_proj(src1)
        b, c, _ = src1.shape
        src1 = src1.reshape(b, c, -1).permute(2, 0, 1)
        src1 = self.norm13(src1)
        q=k=self.with_pos_embed(src1,pos)
        src12 = self.multihead_attn(query=q,
                                     key=k,
                                     value=src1)[0]
        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)
        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)
        src1 = src1.permute(1, 2, 0)

        return src1
    
    