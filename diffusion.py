from functools import partial
from einops import rearrange
from einops.layers.torch import Rearrange
from tqdm import tqdm

from nn_blocks import *


# ===========================================================================
# 时间嵌入
# ===========================================================================

class TimeEmbedding(nn.Module):
    def __init__(self, base_chnl=64, embedding_dim=256):
        super().__init__()
        self.base_chnl = base_chnl
        self.net = nn.Sequential(
            nn.Linear(base_chnl, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, timesteps, **kwargs):
        return self.net(timestep_embedding(timesteps, self.base_chnl))


# ===========================================================================
# 条件编码器: 孔隙度 + 2D 切片 → Cross-Attention 条件序列
# ===========================================================================

class Attention2D(nn.Module):
    def __init__(self, dim=64, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.kv = nn.Conv2d(dim, dim * 2, 1)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, 3, padding=1, groups=dim * 2)
        self.q = nn.Conv2d(dim, dim, 1)
        self.q_dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.project_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x, y):
        b, c, h, w = x.shape
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(x))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return self.project_out(out)


class ConditionEncoder(nn.Module):
    def __init__(self, cond_dim=64):
        super().__init__()
        self.slice_encoder = nn.Sequential(
            nn.Conv2d(1, cond_dim, 3, padding=1), nn.ReLU(),
            nn.Conv2d(cond_dim, cond_dim, 3, padding=1))
        self.porosity_encoder = nn.Sequential(
            nn.Linear(1, cond_dim), nn.GELU(),
            nn.Linear(cond_dim, cond_dim * 2))
        self.positional_encoding = nn.Parameter(torch.zeros(1, 96, cond_dim * 2))
        self.porosity_spatial = nn.Conv1d(cond_dim * 2, cond_dim, 1)

    def forward(self, z_porosities, binary_slice):
        slice_feat = self.slice_encoder(binary_slice)
        # z_porosities: (B, 96) — one scalar per z-slice
        por_emb = self.porosity_encoder(z_porosities.unsqueeze(-1))     # (B, 96, cond_dim*2)
        por_emb = por_emb + self.positional_encoding.to(por_emb.device) # (B, 96, cond_dim*2)
        por_emb = por_emb.permute(0, 2, 1)                             # (B, cond_dim*2, 96)
        por_feat = self.porosity_spatial(por_emb)                       # Conv1d → (B, cond_dim, 96)
        por_feat = por_feat.unsqueeze(-1).expand(-1, -1, -1, binary_slice.shape[-1])  # (B, cond_dim, 96, W)
        return por_feat, slice_feat


class AdaptiveFusionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.porosity_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(4, channels))
        self.slice_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(4, channels))
        self.por_att = Attention2D(dim=channels)
        self.slice_att = Attention2D(dim=channels)
        self.fuse_gate = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 3, padding=1), nn.Sigmoid())
        self.output_conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, por_feat, slice_feat):
        por_enhanced = self.por_att(self.porosity_branch(por_feat), slice_feat) + por_feat
        slice_enhanced = self.slice_att(self.slice_branch(slice_feat), por_feat) + slice_feat
        fuse_map = self.fuse_gate(torch.cat([por_enhanced, slice_enhanced], dim=1))
        fused = fuse_map * por_enhanced + (1 - fuse_map) * slice_enhanced
        return self.output_conv(fused) + fused


class UnifiedConditionEncoder(nn.Module):
    def __init__(self, cond_dim=64):
        super().__init__()
        self.encoder = ConditionEncoder(cond_dim)
        self.fuser = AdaptiveFusionBlock(cond_dim)
        self.to_sequence = Rearrange('b c h w -> b (h w) c')

    def forward(self, porosity, binary_slice):
        por_feat, slice_feat = self.encoder(porosity, binary_slice)
        return self.to_sequence(self.fuser(por_feat, slice_feat))


# ===========================================================================
# 去噪 UNet
# ===========================================================================

class Slice_UNet(nn.Module):
    def __init__(self, *, in_chnl=6, out_chnl=6, base_chnl=64,
                 num_heads=8, chnl_mult=(1, 2, 4), num_res=1, time_dim=256,
                 att_mult=(4, 2, 1), context_dim=256):
        super().__init__()

        self.in_block = TimestepEmbedSequential(conv_nd(3, in_chnl, base_chnl, 3, padding=1))

        ch = base_chnl
        ds = 1
        down_block_ch = [base_chnl]

        self.down_block = nn.ModuleList()
        for level, mult in enumerate(chnl_mult):
            for _ in range(num_res):
                down_layer = [ResBlock_raw(ch, time_dim, out_channel=mult * base_chnl)]
                ch = mult * base_chnl
                if ds in att_mult:
                    down_layer.append(
                        CrossBlock(ch, num_heads=num_heads, dim_head=ch // num_heads, context_dim=context_dim))
                self.down_block.append(TimestepEmbedSequential(*down_layer))
                down_block_ch.append(ch)

            if level != len(chnl_mult) - 1:
                self.down_block.append(TimestepEmbedSequential(Downsample_(ch)))
                down_block_ch.append(ch)
                ds *= 2

        self.mid_block = TimestepEmbedSequential(
            ResBlock_raw(ch, time_dim, out_channel=ch),
            CrossBlock(ch, num_heads=num_heads, dim_head=ch // num_heads, context_dim=context_dim),
            ResBlock_raw(ch, time_dim, out_channel=ch),
        )

        self.up_block = nn.ModuleList()
        for level, mult in reversed(list(enumerate(chnl_mult))):
            for i in range(num_res + 1):
                ich = down_block_ch.pop()
                up_layer = [ResBlock_raw(ch + ich, time_dim, out_channel=mult * base_chnl)]
                ch = base_chnl * mult
                if ds in att_mult:
                    up_layer.append(
                        CrossBlock(ch, num_heads=num_heads, dim_head=ch // num_heads, context_dim=context_dim))
                if level and i == num_res:
                    up_layer.append(Upsample_(ch))
                    ds //= 2
                self.up_block.append(TimestepEmbedSequential(*up_layer))

        self.out_block = nn.Sequential(
            Normalize(base_chnl), nn.SiLU(),
            zero_module(conv_nd(3, base_chnl, out_chnl, 3, padding=1)))

    def forward(self, x, S_emb, context=None, **kwargs):
        hs = []
        h = self.in_block(x, S_emb, context)
        hs.append(h)

        for module in self.down_block:
            h = module(h, S_emb, context)
            hs.append(h)

        h = self.mid_block(h, S_emb, context)

        for module in self.up_block:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, S_emb, context)

        return self.out_block(h)


# ===========================================================================
# 扩散主模型: 时间嵌入 + 条件编码器 + UNet
# ===========================================================================

class SP_Model(nn.Module):
    def __init__(self, *, in_chnl=6, out_chnl=6, base_chnl=64,
                 num_heads=8, chnl_mult=(1, 2, 4), num_res=1, time_dim=256,
                 att_mult=(4, 2, 1), context_dim=256,
                 base_chnl_emb=64, embedding_dim=256, cond_dim=256):
        super().__init__()
        self.SNet = Slice_UNet(in_chnl=in_chnl, out_chnl=out_chnl, base_chnl=base_chnl,
                               num_heads=num_heads, chnl_mult=chnl_mult, num_res=num_res,
                               time_dim=time_dim, att_mult=att_mult, context_dim=context_dim)
        self.Emb = TimeEmbedding(base_chnl=base_chnl_emb, embedding_dim=embedding_dim)
        self.CondEncoder = UnifiedConditionEncoder(cond_dim=cond_dim)

    def forward(self, x, timesteps, context, slice_, **kwargs):
        cond = self.CondEncoder(context, slice_)
        S_emb = self.Emb(timesteps)
        return self.SNet(x, S_emb, cond)


# ===========================================================================
# DDPM 扩散过程
# ===========================================================================

class DDPM_(nn.Module):
    def __init__(self, *, sfnet: SP_Model, timesteps=1000, image_size=12, channels=6,
                 linear_start=0.0015, linear_end=0.0195):
        super().__init__()
        self.image_size = image_size
        self.channels = channels
        self.model = sfnet
        count_params(self.model, verbose=True)

        self.register_schedule(timesteps=timesteps, linear_start=linear_start, linear_end=linear_end)

    def register_schedule(self, timesteps=1000, linear_start=0.0015, linear_end=0.0195):
        betas = make_beta_schedule("linear", timesteps, linear_start=linear_start, linear_end=linear_end)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.num_timesteps = int(timesteps)
        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped',
                             to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1',
                             to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2',
                             to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def predict_start_from_noise(self, x_t, t, noise):
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise)

    def q_posterior_mean_variance(self, x_start, x_t, t):
        mean = (extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        log_var = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, log_var

    @torch.no_grad()
    def p_sample(self, x, t, context=None, slice_=None):
        b, device = x.shape[0], x.device
        model_out = self.model(x=x, timesteps=t, context=context, slice_=slice_)
        x_recon = self.predict_start_from_noise(x, t, model_out).clamp(-1., 1.)
        mean, log_var = self.q_posterior_mean_variance(x_recon, x, t)
        noise = torch.randn_like(x)
        mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return mean + mask * (0.5 * log_var).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, context=None, slice_=None):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        for i in tqdm(reversed(range(self.num_timesteps)), desc='Sampling', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                context=context, slice_=slice_)
        return img

    @torch.no_grad()
    def sample(self, batch_size, context=None, slice_=None):
        return self.p_sample_loop(
            (batch_size, self.channels, self.image_size, self.image_size, self.image_size),
            context=context, slice_=slice_)

    def p_losses(self, x_start, t, context=None, slice_=None):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        model_out = self.model(x=x_noisy, timesteps=t, context=context, slice_=slice_)
        loss = (noise - model_out).abs().mean(dim=[1, 2, 3, 4]).mean()
        return loss, {"loss": loss.detach()}

    def forward(self, x, device, context=None, slice_=None):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=device).long()
        return self.p_losses(x, t, context=context, slice_=slice_)
