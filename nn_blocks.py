from abc import abstractmethod

import torch.nn.functional as F
from einops import rearrange

from nn_utils import *


# ---------------------------------------------------------------------------
# 上下采样 (3D)
# ---------------------------------------------------------------------------

class Upsample_(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = conv_nd(3, channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class Downsample_(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = conv_nd(3, channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------

class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        pass


# ---------------------------------------------------------------------------
# VAE 残差块
# ---------------------------------------------------------------------------

class ResnetBlock_vae(nn.Module):
    def __init__(self, in_channel, out_channel=None):
        super().__init__()
        out_channel = out_channel or in_channel
        self.norm1 = Normalize(in_channel)
        self.conv1 = conv_nd(3, in_channel, out_channel, 3, padding=1)
        self.norm2 = Normalize(out_channel)
        self.conv2 = conv_nd(3, out_channel, out_channel, 3, padding=1)
        self.skip = conv_nd(3, in_channel, out_channel, 1) if in_channel != out_channel else nn.Identity()

    def forward(self, x):
        h = nonlinearity(self.norm1(x))
        h = self.conv1(h)
        h = nonlinearity(self.norm2(h))
        h = self.conv2(h)
        return self.skip(x) + h


# ---------------------------------------------------------------------------
# UNet 残差块 (含时间嵌入注入)
# ---------------------------------------------------------------------------

class ResBlock_raw(TimestepBlock):
    def __init__(self, channel, emb_channel, out_channel=None):
        super().__init__()
        out_channel = out_channel or channel
        self.in_layers = nn.Sequential(Normalize(channel), nn.SiLU(), conv_nd(3, channel, out_channel, 3, padding=1))
        self.emb_layers = nn.Sequential(nn.SiLU(), linear(emb_channel, out_channel))
        self.out_layers = nn.Sequential(Normalize(out_channel), nn.SiLU(),
                                        zero_module(conv_nd(3, out_channel, out_channel, 3, padding=1)))
        self.skip_connection = conv_nd(3, channel, out_channel, 1) if out_channel != channel else nn.Identity()

    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h

    def forward(self, x, emb):
        return checkpoint(self._forward, (x, emb), self.parameters(), True)


# ---------------------------------------------------------------------------
# Cross-Attention
# ---------------------------------------------------------------------------

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim
        self.net = nn.Sequential(GEGLU(dim, inner_dim), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, num_heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * num_heads
        context_dim = context_dim or query_dim
        self.scale = dim_head ** -0.5
        self.heads = num_heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None):
        h = self.heads
        context = context if context is not None else x
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class BasicCrossBlock(nn.Module):
    def __init__(self, query_dim, num_heads, dim_head, context_dim=None):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=query_dim, num_heads=num_heads, dim_head=dim_head)
        self.attn2 = CrossAttention(query_dim=query_dim, context_dim=context_dim, num_heads=num_heads,
                                    dim_head=dim_head)
        self.ff = FeedForward(query_dim)
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.norm3 = nn.LayerNorm(query_dim)

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class CrossBlock(nn.Module):
    """将 3D 特征序列化，过 transformer，再还原为 3D。"""

    def __init__(self, in_channels, num_heads=4, dim_head=32, context_dim=None):
        super().__init__()
        inner_dim = num_heads * dim_head
        self.norm = Normalize(in_channels)
        self.proj_in = conv_nd(3, in_channels, inner_dim, 1)
        self.transformer = BasicCrossBlock(inner_dim, num_heads, dim_head, context_dim=context_dim)
        self.proj_out = zero_module(conv_nd(3, inner_dim, in_channels, 1))

    def _forward(self, x, context=None):
        b, c, h, w, d = x.shape
        x_in = x
        x = self.proj_in(self.norm(x))
        x = rearrange(x, 'b c h w d -> b (h w d) c')
        x = self.transformer(x, context=context)
        x = rearrange(x, 'b (h w d) c -> b c h w d', h=h, w=w, d=d)
        return self.proj_out(x) + x_in

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), True)


# ---------------------------------------------------------------------------
# 参数分发容器
# ---------------------------------------------------------------------------

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, CrossBlock):
                x = layer(x, context)
            else:
                x = layer(x)
        return x
