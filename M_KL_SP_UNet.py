# -*- coding: utf-8 -*-
# 上面是以前的↑ #
from F_fct_module import *
from B_Res import ResBlock_raw
from B_UpDn import Upsample_, Downsample_
from B_Att import make_attn
from B_Emb import SF_emb
from Block_forward import TimestepEmbedSequential
from B_SP_Cross import UnifiedConditionEncoder


class Slice_UNet(nn.Module):

    def __init__(
            self, *,
            dim,
            in_chnl,
            out_chnl,
            base_chnl,

            # input
            num_cls,
            # Att
            num_heads,
            # num_heads = ch // num_head_channels (num_head_channels==dim head)          if num_heads_upsample == -1: num_heads_upsample = num_heads
            use_fp16,
            att_mult,
            use_checkpoint_att=True,
            norm=True,
            context_dim=None,
            # Res
            num_res,
            chnl_mult,
            time_dim,
            drop_out=0,
            use_scale_shift_norm=False,
            use_checkpoint_res=True,
            # UD_sample
            res_updown=False,
            use_checkpoint_ud=True,
            use_conv=True,
            # High Frequency Augment
            use_highfrq=True,
            **ignore_kwargs
    ):
        super().__init__()

        # # 引导信息 cross attention 暂时不用 但是保留
        # if use_spatial_transformer:
        #     assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'
        #
        # if context_dim is not None:
        #     assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
        #     from omegaconf.listconfig import ListConfig
        #     if type(context_dim) == ListConfig:
        #         context_dim = list(context_dim)

        # self #
        self.num_classes = num_cls
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.use_highfrq = False


#         self.time_embed = nn.Sequential(
#             linear(model_channels, time_embed_dim),
#             nn.SiLU(),
#             linear(time_embed_dim, time_embed_dim),
#         )

#         if self.num_classes is not None:
#             self.label_emb = nn.Embedding(num_classes, time_embed_dim)



        # assert #
#         assert time_dim == base_chnl or time_dim == base_chnl*4

        # in_block #
        self.in_block = TimestepEmbedSequential(conv_nd(dim, in_chnl, base_chnl, 3, padding=1))

        # down_block #
        self.down_block = nn.ModuleList([])

        # ch_register
        ch = base_chnl
        ds = 1
        down_block_ch = [base_chnl]

        for level, mult in enumerate(chnl_mult):  # enumerate同时返回索引和对应的值

            # Res + Att
            for _ in range(num_res):
                # Res
                down_layer = [
                    ResBlock_raw(
                        ch,
                        time_dim,
                        drop_out,
                        out_channel=mult * base_chnl,
                        dims=dim,
                        use_checkpoint=use_checkpoint_res,
                    )
                ]

                # ch...
                ch = mult * base_chnl

                # Att
                if ds in att_mult:
                    dim_head = ch // num_heads
                    down_layer.append(
#                         make_attn(
#                             dim,
#                             ch,
#                             use_checkpoint_att,
#                             norm=True,
#                             num_heads=num_heads,
#                             dim_head=dim_head,
#                             attn_type="basic_" if level!=0 else 'linear',
#                         )
                        make_attn(
                            dim,
                            ch,
                            use_checkpoint_att,
                            norm=True,
                            num_heads=num_heads,
                            dim_head=dim_head,
                            attn_type="cross",
                            context_dim=context_dim,
                        )
                    )
                else:
                    print(f'level {level}:No Atten')

                # block append
                self.down_block.append(TimestepEmbedSequential(*down_layer))

                # ch...
                down_block_ch.append(ch)

            # DownSample
            if level != len(chnl_mult) - 1:
                # block append
                self.down_block.append(
                    TimestepEmbedSequential(
                        ResBlock_raw(
                            ch,
                            time_dim,
                            drop_out,
                            out_channel=ch,
                            dims=dim,
                            use_checkpoint=use_checkpoint_ud,
                            use_scale_shift_norm=True,
                            up=False,
                            down=True,
                        ) if res_updown else Downsample_(ch, use_conv, dims=dim, out_channels=ch, padding=1)
                    )
                )
                # ch...
                down_block_ch.append(ch)
                ds *= 2

        # mid_block #

        # ch_register
        dim_head = ch // num_heads

        self.mid_block = TimestepEmbedSequential(
            ResBlock_raw(
                ch,
                time_dim,
                drop_out,
                out_channel=ch,
                dims=dim,
                use_checkpoint=use_checkpoint_res,
            ),
#             make_attn(
#                 dim,
#                 ch,
#                 use_checkpoint_att,
#                 norm=True,
#                 num_heads=num_heads,
#                 dim_head=dim_head,
#                 attn_type="basic_" if level!=0 else 'linear',
#             )
            make_attn(
                dim,
                ch,
                use_checkpoint_att,
                norm=True,
                num_heads=num_heads,
                dim_head=dim_head,
                attn_type="cross",
                context_dim=context_dim,
            )              
            ,
            ResBlock_raw(
                ch,
                time_dim,
                drop_out,
                out_channel=ch,
                dims=dim,
                use_checkpoint=use_checkpoint_res,
            ),
        )

        # up_block #
        self.up_block = nn.ModuleList([])
        for level, mult in list(enumerate(chnl_mult))[::-1]:
            for i in range(num_res + 1):
                # pop to cat
                ich = down_block_ch.pop()

                # Res
                up_layer = [
                    # channel,
                    # emb_channel,
                    # dropout,
                    # out_channel = None,
                    # use_conv = False,
                    # use_scale_shift_norm = False,
                    # dims = 3,
                    # use_checkpoint = False,
                    ResBlock_raw(
                        ch + ich,
                        time_dim,
                        drop_out,
                        out_channel=mult * base_chnl,
                        dims=dim,
                        use_checkpoint=use_checkpoint_res,
                    )
                ]

                # ch...
                ch = base_chnl * mult

                # Att
                if ds in att_mult:
                    dim_head = ch // num_heads
                    up_layer.append(
#                         make_attn(
#                             dim,
#                             ch,
#                             use_checkpoint_att,
#                             norm=True,
#                             num_heads=num_heads,
#                             dim_head=dim_head,
#                             attn_type="basic_" if level!=0 else 'linear',
#                         )
                        make_attn(
                            dim,
                            ch,
                            use_checkpoint_att,
                            norm=True,
                            num_heads=num_heads,
                            dim_head=dim_head,
                            attn_type="cross",
                            context_dim=context_dim,
                        )  
                    )
                else:
                    print(f'level {level}:No Atten')

                # DownSample
                if level and i == num_res:
                    up_layer.append(
                        ResBlock_raw(
                            ch,
                            time_dim,
                            drop_out,
                            out_channel=ch,
                            dims=dim,
                            use_checkpoint=use_checkpoint_ud,
                            use_scale_shift_norm=True,
                            up=True,
                            down=False,
                        ) if res_updown else Upsample_(ch, use_conv, dims=dim, out_channels=ch, padding=1)
                    )

                    # in_block
                    ds //= 2

                # block append
                self.up_block.append(TimestepEmbedSequential(*up_layer))

        # hfa_block High Frequency Augment #
        if self.use_highfrq:
            self.hfa_block = HighFreqEnhancer(ch, dims=dim)

        # out_block #
        self.out_block = nn.Sequential(
            Normalize(base_chnl),
            nn.SiLU(),
            zero_module(conv_nd(dim, base_chnl, out_chnl, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.in_block.apply(convert_module_to_f16)
        self.down_block.apply(convert_module_to_f16)
        self.mid_block.apply(convert_module_to_f16)
        self.up_block.apply(convert_module_to_f16)
        self.hfa_block.apply(convert_module_to_f16)
        self.out_block.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.in_block.apply(convert_module_to_f32)
        self.down_block.apply(convert_module_to_f32)
        self.mid_block.apply(convert_module_to_f32)
        self.up_block.apply(convert_module_to_f32)
        self.hfa_block.apply(convert_module_to_f32)
        self.out_block.apply(convert_module_to_f32)

    def forward(self, x, S_emb, context=None, label=-1, class_ids=-1, **kwargs):

#         assert (y is not None) == (self.num_classes is not None), "must specify y if and only if the model is class-conditional"
#         t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
#         emb = self.time_embed(t_emb)

#         if self.num_classes is not None:
#             assert y.shape == (x.shape[0],)
#             emb = emb + self.label_emb(y)


        h = x.type(self.dtype)
        
        
        assert S_emb != None

        hs = []
        h = x.type(self.dtype)

        # in #
        h = self.in_block(h, S_emb)
        hs.append(h)

        # down #
#         down = 0
#         print(f'down:{down}')
        for module in self.down_block:
            h = module(h, S_emb, context, label, class_ids)
#             down+=1
#             print(f'down:{down} //h:{h.shape} //module class: {module.__class__.__name__}')
            # assert torch.isfinite(h).all(), "h contains NaN or inf"
            hs.append(h)

        # mid #
        h = self.mid_block(h, S_emb, context, label, class_ids)
        # assert torch.isfinite(h).all(), "h contains NaN or inf"

        # up #
#         up=0
#         print(f'up:{up}')
        for module in self.up_block:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, S_emb, context, label, class_ids)
#             up+=1
#             print(f'up:{up} //h:{h.shape} //module class: {module.__class__.__name__}')
            # assert torch.isfinite(h).all(), "h contains NaN or inf"

        # hfa #
        if self.use_highfrq:
            h = self.hfa_block(h, S_emb)

        # out #
        h = self.out_block(h)

        return h

    


class SP_Model(nn.Module):

    def __init__(
            self,
            *,
            S_cfg=None,
            Emb_cfg=None,
            **ignore_kwargs
    ):
        super().__init__()
        
        S_cfg['context_dim'] = S_cfg['base_chnl'] * (2 **( len(S_cfg['chnl_mult']) -1 ))
        assert S_cfg['context_dim'] % S_cfg['base_chnl'] == 0,f"context_dim:{context_dim}  base_chnl:{S_cfg['base_chnl']}"
        context_dim = S_cfg['context_dim']
        max_slices = S_cfg['max_slices']
        learnable_pos = S_cfg['learnable_pos']
        print(f"context_dim:{context_dim}  base_chnl:{S_cfg['base_chnl']}  max_slices:{max_slices} learnable_pos:{S_cfg['learnable_pos']}")
        
        self.SNet = Slice_UNet(**S_cfg)
        self.Emb = SF_emb(**Emb_cfg)
        
        self.CondEncoder = UnifiedConditionEncoder(cond_dim=context_dim,learnable_pos=learnable_pos)

    def forward(self, x, timesteps=None, y=None, context=None, slice_=None ,**kwargs):
        cond = self.CondEncoder(context,slice_)

        S_emb, F_emb = self.Emb(x, timesteps=timesteps, y=y, **kwargs)

        S_out = self.SNet(x, S_emb=S_emb, context=cond, **kwargs)

        return S_out, -1
