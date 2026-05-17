from F_fct_module import *
from B_UpDn import Upsample_,Downsample_
from abc import abstractmethod


class ResnetBlock_vae(nn.Module):
    def __init__(self, *, dims, in_channel, out_channel=None, conv_shortcut=False,
                 dropout=0., temb_channel=0, use_checkpoint=False):
        super().__init__()

        self.in_channel = in_channel
        out_channel = in_channel if out_channel is None else out_channel
        self.out_channel = out_channel
        self.use_conv_shortcut = conv_shortcut
        self.use_checkpoint = use_checkpoint

        self.norm1 = Normalize(in_channel)
        self.conv1 = conv_nd(dims, in_channel, out_channel, kernel_size=3, stride=1, padding=1)

        if temb_channel > 0:
            self.emb_layers = nn.Sequential(
                nn.Linear(temb_channel,out_channel),
                nn.SiLU(),
            )
    

        self.norm2 = Normalize(out_channel)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = conv_nd(dims, out_channel, out_channel, kernel_size=3, stride=1, padding=1)

        if self.in_channel != self.out_channel:
            if self.use_conv_shortcut:
                self.conv_shortcut = conv_nd(dims, in_channel, out_channel, kernel_size=3, stride=1,padding=1)
            else:
                self.nin_shortcut = conv_nd(dims, in_channel, out_channel, kernel_size=1, stride=1,padding=0)

    def _forward(self, x, temb):
        '''
        Args:
        Returns:
            channel_cut{  Conv[Drop[Sig[GN[ Conv(Sig(GN(x))) ]]]]  }
        '''
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            emb_out = self.emb_layers(temb).type(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
            h = h + emb_out


        if self.in_channel != self.out_channel:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channel] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    
    
    
    
    


#
class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """



#
class ResBlock_mini(TimestepBlock):
    """
    A residual block that can optionally change the number of channel.
    :param channel: the number of input channel.
    :param emb_channel: the number of timestep embedding channel.
    :param dropout: the rate of dropout.
    :param out_channel: if specified, the number of out channel.
    :param use_conv: if True and out_channel is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channel in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
            self,
            channel,
            emb_channel,
            dropout,
            out_channel=None,
            use_conv=False,
            use_scale_shift_norm=False,
            dims=3,
            use_checkpoint=False,
    ):
        super().__init__()
        self.channel = channel
        self.emb_channel = emb_channel
        self.dropout = dropout
        self.out_channel = out_channel or channel
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            Normalize(channel),
            nn.SiLU(),
            conv_nd(dims, channel, self.out_channel, 3, padding=1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channel,
                2 * self.out_channel if use_scale_shift_norm else self.out_channel,
            ),
        )
        self.out_layers = nn.Sequential(
            Normalize(self.out_channel),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channel, self.out_channel, 3, padding=1)
            ),
        )

        if self.out_channel == channel:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channel, self.out_channel, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channel, self.out_channel, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channel] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):

        h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)

        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)

        else:
            h = h + emb_out
            h = self.out_layers(h)

        h = self.skip_connection(x) + h
        assert torch.isfinite(h).all(), "h = self.skip_connection(x) + h contains NaN or inf"

        return h



class ResBlock_raw(TimestepBlock):
    """
    A residual block that can optionally change the number of channel.
    :param channel: the number of input channel.
    :param emb_channel: the number of timestep embedding channel.
    :param dropout: the rate of dropout.
    :param out_channel: if specified, the number of out channel.
    :param use_conv: if True and out_channel is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channel in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channel,
        emb_channel,
        dropout,
        out_channel=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=3,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        assert (up==use_scale_shift_norm) or (down==use_scale_shift_norm),\
            '当使用ResBlock作为上下采样模块(up/dowm==True),一般设定use_scale_shift_norm==True'
        self.channel = channel
        self.emb_channel = emb_channel
        self.dropout = dropout
        self.out_channel = out_channel or channel
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            Normalize(channel),
            nn.SiLU(),
            conv_nd(dims, channel, self.out_channel, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample_(channel, False, dims)
            self.x_upd = Upsample_(channel, False, dims)
        elif down:
            self.h_upd = Downsample_(channel, False, dims)
            self.x_upd = Downsample_(channel, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()
            

#         S_emb:torch.Size([3, 64]), F_emb:torch.Size([3, 64])
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channel,
                2 * self.out_channel if use_scale_shift_norm else self.out_channel,
            ),
        )
        self.out_layers = nn.Sequential(
            Normalize(self.out_channel),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channel, self.out_channel, 3, padding=1)
            ),
        )

        if self.out_channel == channel:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channel, self.out_channel, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channel, self.out_channel, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channel] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )


    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        h = self.skip_connection(x) + h
        assert torch.isfinite(h).all(), "h = self.skip_connection(x) + h contains NaN or inf"
        return h

