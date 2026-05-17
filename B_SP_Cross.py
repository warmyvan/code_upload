import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


class Attention(nn.Module):
    def __init__(self, dim=64, num_heads=8, bias=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv2d(dim, dim , kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(x))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
    

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class ConditionEncoder(nn.Module):

    def __init__(self, cond_dim, learnable_pos):
        super().__init__()

        self.slice_encoder = nn.Sequential(
            nn.Conv2d(1, cond_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(cond_dim, cond_dim, 3, padding=1)
        )

        self.porosity_encoder = nn.Sequential(
            nn.Linear(1, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim*2),
        )

        max_slices=96
        if learnable_pos:
            self.positional_encoding = nn.Parameter(torch.zeros(1, max_slices, cond_dim*2))
        else:
            self.positional_encoding = self._generate_sinusoidal_embeddings(max_slices, cond_dim*2)
        self.porosity_spatial = nn.Conv1d(cond_dim*2, cond_dim, 1)

    def forward(self, porosity, binary_slice):
        slice_feat = self.slice_encoder(binary_slice)

        porosity = porosity.unsqueeze(-1)
        por_emb = self.porosity_encoder(porosity) 
        por_emb = por_emb + self.positional_encoding.to(por_emb.device)
        por_emb = por_emb.permute(0, 2, 1)
        por_feat = self.porosity_spatial(por_emb)
        por_feat = por_feat.unsqueeze(-1).expand(-1, -1, -1, (binary_slice.shape[-1]))

        return por_feat, slice_feat
    
    def _generate_sinusoidal_embeddings(self, length, dim):
        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(torch.log(torch.tensor(10000.0)) / dim))
        pe = torch.zeros(1, length, dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe


class AdaptiveFusionBlock(nn.Module):
    """改进的融合模块，处理异构输入"""

    def __init__(self, channels):
        super().__init__()

        self.porosity_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(4, channels)
        )

        self.slice_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(4, channels)
        )

        self.por_att = Attention(dim=channels)
        self.slice_att = Attention(dim=channels)

        self.fuse_gate = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 3, padding=1),
            nn.Sigmoid()
        )

        self.output_conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, por_feat, slice_feat):
        por_global = self.porosity_branch(por_feat)
        slice_local = self.slice_branch(slice_feat)
        
        por_enhanced = self.por_att(por_global, slice_local) + por_feat
        slice_enhanced = self.slice_att(slice_local, por_global) + slice_feat

        fuse_map = self.fuse_gate(torch.cat([por_enhanced, slice_enhanced], dim=1))
        fused = fuse_map * por_enhanced + (1 - fuse_map) * slice_enhanced

        return self.output_conv(fused) + fused


class UnifiedConditionEncoder(nn.Module):

    def __init__(self,cond_dim=64, learnable_pos=True):
        super().__init__()
        self.encoder = ConditionEncoder(
            cond_dim=cond_dim,
            learnable_pos=learnable_pos
        )
        self.fuser = AdaptiveFusionBlock(cond_dim)

        self.to_sequence = nn.Sequential(
            Rearrange('b c h w -> b (h w) c')
        )

    def forward(self, porosity, binary_slice):
        por_feat, slice_feat = self.encoder(porosity, binary_slice)
        fused = self.fuser(por_feat, slice_feat)
        return self.to_sequence(fused)

