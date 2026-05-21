from nn_blocks import *


class DiagonalGaussianDistribution:
    def __init__(self, parameters):
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self):
        return self.mean + self.std * torch.randn_like(self.mean)

    def kl(self):
        return 0.5 * torch.sum(self.mean.pow(2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3, 4])


class Encoder(nn.Module):
    def __init__(self, ch=64, ch_mult=(1, 2, 4, 8), num_res_blocks=1,
                 in_channels=1, z_channels=4):
        super().__init__()
        self.conv_in = conv_nd(3, in_channels, ch, 3, padding=1)

        self.down = nn.ModuleList()
        in_ch_mult = (1,) + tuple(ch_mult)

        for i_level in range(len(ch_mult)):
            block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                block.append(ResnetBlock_vae(block_in, block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            if i_level != len(ch_mult) - 1:
                down.downsample = Downsample_(block_in)
            self.down.append(down)

        self.mid_block_1 = ResnetBlock_vae(block_in, block_in)
        self.mid_block_2 = ResnetBlock_vae(block_in, block_in)

        self.norm_out = Normalize(block_in)
        self.conv_out = conv_nd(3, block_in, 2 * z_channels, 3, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        for i_level in range(len(self.down)):
            for block in self.down[i_level].block:
                h = block(h)
            if hasattr(self.down[i_level], 'downsample'):
                h = self.down[i_level].downsample(h)
        h = self.mid_block_1(h)
        h = self.mid_block_2(h)
        h = nonlinearity(self.norm_out(h))
        return self.conv_out(h)


class Decoder(nn.Module):
    def __init__(self, ch=64, out_ch=1, ch_mult=(1, 2, 4, 8), num_res_blocks=1, z_channels=4):
        super().__init__()
        block_in = ch * ch_mult[-1]

        self.conv_in = conv_nd(3, z_channels, block_in, 3, padding=1)

        self.mid_block_1 = ResnetBlock_vae(block_in, block_in)
        self.mid_block_2 = ResnetBlock_vae(block_in, block_in)

        self.up = nn.ModuleList()
        for i_level in reversed(range(len(ch_mult))):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks + 1):
                block.append(ResnetBlock_vae(block_in, block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = Upsample_(block_in)
            self.up.insert(0, up)

        self.norm_out = Normalize(block_in)
        self.conv_out = conv_nd(3, block_in, out_ch, 3, padding=1)

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid_block_1(h)
        h = self.mid_block_2(h)
        for i_level in reversed(range(len(self.up))):
            for block in self.up[i_level].block:
                h = block(h)
            if hasattr(self.up[i_level], 'upsample'):
                h = self.up[i_level].upsample(h)
        h = nonlinearity(self.norm_out(h))
        return self.conv_out(h)


class AutoencoderKL_(nn.Module):
    def __init__(self, *, ch=64, ch_mult=(1, 2, 4, 8), num_res_blocks=1,
                 in_channels=1, out_ch=1, z_channels=4, embed_dim=6,
                 kl_weight=1e-6, learning_rate=1e-5):
        super().__init__()
        self.kl_weight = kl_weight
        self.learning_rate = learning_rate

        self.encoder = Encoder(ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               in_channels=in_channels, z_channels=z_channels)
        self.decoder = Decoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult,
                               num_res_blocks=num_res_blocks, z_channels=z_channels)

        self.quant_conv = conv_nd(3, 2 * z_channels, 2 * embed_dim, 1)
        self.post_quant_conv = conv_nd(3, embed_dim, z_channels, 1)

    def encode(self, x):
        moments = self.quant_conv(self.encoder(x))
        return DiagonalGaussianDistribution(moments)

    def decode(self, z):
        return self.decoder(self.post_quant_conv(z))

    def forward(self, x):
        posterior = self.encode(x)
        z = posterior.sample()
        dec = self.decode(z)
        return dec, posterior

    def get_loss(self, inputs, reconstructions, posterior):
        rec_loss = torch.abs(inputs - reconstructions)
        nll_loss = torch.sum(rec_loss) / rec_loss.shape[0]
        kl_loss = posterior.kl().mean()
        loss = nll_loss + self.kl_weight * kl_loss
        return loss, {"loss": loss.detach(), "rec_loss": rec_loss.detach().mean(), "kl_loss": kl_loss.detach()}

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.quant_conv.parameters()) +
            list(self.post_quant_conv.parameters()),
            lr=self.learning_rate, betas=(0.5, 0.9))
        return opt
