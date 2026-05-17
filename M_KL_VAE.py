from F_fct_module import *
from B_Res import ResBlock_mini,ResnetBlock_vae
from B_Att import make_attn
from B_UpDn import Upsample_,Downsample_
from B_Spct import SpectralConvnd_SR
from L_loss_raw import LPIPSWithDiscriminator,DiagonalGaussianDistribution


class Encoder(nn.Module):
    def __init__(self, *, dims, ch, use_checkpoint ,ch_mult=(1, 2, 4, 8), num_res_blocks, label_embed_dim,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="basic",
                 norm=True, num_heads=8, dim_head=32,
                 **ignore_kwargs):
        super().__init__()

#         assert (attn_type == "linear")and(use_linear_attn==True),'cfg: atten type error'
        self.ch = ch
        self.temb_ch = label_embed_dim
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.dims = dims

        self.conv_in = conv_nd(dims, in_channels, self.ch, kernel_size=3, stride=1,padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]

            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock_vae(
                        dims=dims,
                        in_channel=block_in,
                        out_channel=block_out,
                        temb_channel=label_embed_dim,
                        dropout=dropout,
                        use_checkpoint=use_checkpoint,
                    )
                )

                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(dims, block_in, use_checkpoint, attn_type=attn_type,norm=norm, num_heads=num_heads, dim_head=dim_head))

            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample_(block_in, resamp_with_conv, dims=dims)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock_vae(dims=self.dims,
                                           in_channel=block_in,
                                           out_channel=block_in,
                                           temb_channel=self.temb_ch,
                                           dropout=dropout,
                                           use_checkpoint=use_checkpoint)

        self.mid.attn_1 = make_attn(dims, block_in, use_checkpoint, attn_type=attn_type,norm=norm, num_heads=num_heads, dim_head=dim_head)
        self.mid.block_2 = ResnetBlock_vae(dims=self.dims,
                                           in_channel=block_in,
                                           out_channel=block_in,
                                           temb_channel=self.temb_ch,
                                           dropout=dropout,
                                           use_checkpoint=use_checkpoint)


        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = conv_nd(self.dims,
                                block_in,
                                2 * z_channels if double_z else z_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1)

    def forward(self, x, label):
        # timestep embedding
        temb = label

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


# self.decoder = Decoder(**ENconfig)
class Decoder(nn.Module):
    def __init__(self, *, dims, ch, out_ch, use_checkpoint,ch_mult=(1, 2, 4, 8), num_res_blocks, 
                 label_embed_dim,attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,attn_type="linear",
                 norm=True, num_heads=8, dim_head=32,
                  **ignorekwargs):  # give_pre_end=False, tanh_out=False,
        super().__init__()
        if use_linear_attn: 
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = label_embed_dim
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.dims = dims

        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        self.conv_in = conv_nd(self.dims,
                               z_channels,
                               block_in,
                               kernel_size=3,
                               stride=1,
                               padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock_vae(dims=self.dims,
                                       in_channel=block_in,
                                       out_channel=block_in,
                                       temb_channel=self.temb_ch,
                                       dropout=dropout,
                                       use_checkpoint=use_checkpoint,)

        self.mid.attn_1 = make_attn(dims, block_in, use_checkpoint, attn_type=attn_type,norm=norm, num_heads=num_heads, dim_head=dim_head)
        self.mid.block_2 = ResnetBlock_vae(dims=self.dims,
                                       in_channel=block_in,
                                       out_channel=block_in,
                                       temb_channel=self.temb_ch,
                                       dropout=dropout,
                                       use_checkpoint = use_checkpoint,)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock_vae(dims=self.dims,
                                         in_channel=block_in,
                                         out_channel=block_out,
                                         temb_channel=self.temb_ch,
                                         dropout=dropout,
                                         use_checkpoint = use_checkpoint,))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(dims, block_in, use_checkpoint, attn_type=attn_type,norm=norm, num_heads=num_heads, dim_head=dim_head))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample_(block_in, resamp_with_conv,dims=dims)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)

        self.conv_out = conv_nd(self.dims,
                                block_in,
                                out_ch,
                                kernel_size=3,
                                stride=1,
                                padding=1)

    def forward(self, z, label):
        self.last_z_shape = z.shape
        temb = label

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h




class AutoencoderKL_(nn.Module):
    def __init__(self,
                 epoch,
                 embed_dim,
                 ENconfig,
                 # DEconfig,
                 lossconfig,
                 ):
        super().__init__()
        
        # img “nn.Embedding.png” detail
        if ENconfig['label']==True: 
            self.label_emb = nn.Embedding(ENconfig['num_labels'], ENconfig['label_embed_dim'])

        self.global_step = epoch
        self.encoder = Encoder(**ENconfig)
        self.decoder = Decoder(**ENconfig)
        self.loss = LPIPSWithDiscriminator(**lossconfig)

        assert ENconfig["double_z"]
        
        
        dims = ENconfig['dims']
        self.dims = dims
        self.quant_conv = conv_nd(dims, 2 * ENconfig["z_channels"], 2 * embed_dim, kernel_size=1 )
        self.post_quant_conv = conv_nd(dims, embed_dim, ENconfig["z_channels"],  kernel_size=1)
        self.embed_dim = embed_dim

    def encode(self, x, label):
        h = self.encoder(x, label)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments,dims=self.dims)
        return posterior


    def decode(self, z, label):
        z = self.post_quant_conv(z)
        dec = self.decoder(z, label)
        return dec


    def forward(self, input_, label ,sample_posterior=True,clamp=False):

        
        if label[0] > 0:
            label = self.label_emb(label)
        else:
            label = None

        posterior = self.encode(input_, label)
        if sample_posterior:
            z = posterior.sample()

        else:
            z = posterior.mode()
        dec = self.decode(z, label)
        
        if clamp:
            dec = torch.clamp(dec, min=0.0, max=1.0)
        return dec, posterior, z


    # data loader
    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x


    # trainer
    def get_loss(self, inputs, reconstructions, posterior, loss_choice, cond):

        if loss_choice == 0 :
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior,
                                            loss_choice,
                                            global_step=self.global_step,
                                            last_layer=self.get_last_layer(),
                                            cond=cond,
                                            split="train")
            return aeloss, log_dict_ae

        elif loss_choice == 1:
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior,
                                                loss_choice,
                                                global_step=self.global_step,
                                                last_layer=self.get_last_layer(),
                                                cond=cond,
                                                split="train")

            return discloss
        else :
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior,
                                            loss_choice,
                                            global_step=self.global_step,
                                            last_layer=self.get_last_layer(),
                                            cond=cond,
                                            split="train")
            return aeloss, log_dict_ae



    # opt
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []


    def get_last_layer(self):
        return self.decoder.conv_out.weight

    
    def convert_to_fp16(self):
        
        """
        Convert the torso of the model to float16.
        """
        self.label_emb.apply(convert_module_to_f16)
        self.encoder.apply(convert_module_to_f16)
        self.decoder.apply(convert_module_to_f16)
        self.quant_conv.apply(convert_module_to_f16)
        self.post_quant_conv.apply(convert_module_to_f16)

