from F_fct_module import *
from einops import rearrange, repeat
import math

'''
公共初始化参数
    def __init__(self, dims, in_channels, norm=True, use_checkpoint=True, num_heads=1, dim_head=-1, )
        目前dims参数可选择2/3，但是并不能完全适配dims==2的情况:conv_nd可以，但是注意力机制的dhw维度没有改好

    3DSwinTransformer还没有适用
'''


# 


class SimAM(torch.nn.Module):
    def __init__(self, dims, in_channels=None, norm=False, use_checkpoint=False, num_heads=-1, dim_head=-1,
                 e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.SiLU()
        self.e_lambda = e_lambda
        self.dims = dims
        if norm:
            self.norm = Normalize(in_channels)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        if self.dims == 3:
            b, c, h, w, d = x.size()
        else:
            b, c, h, w = x.size()
            d = 1.

        # new add
        if self.norm == True:
            x = self.norm(x)

        if self.dims == 3:
            n = w * h * d - 1
        elif self.dims == 2:
            n = w * h - 1

        if self.dims == 3:
            x_minus_mu_square = (x - x.mean(dim=[2, 3, 4], keepdim=True)).pow(2)
            y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3, 4], keepdim=True) / n + self.e_lambda)) + 0.5
        else:
            x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
            y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


# Axial attention
class SelfAttention_Axial(nn.Module):

    def __init__(self, dims, inchannel, heads, dim_heads=None, ):
        super().__init__()
        self.dim_heads = (inchannel // heads) if dim_heads is None else dim_heads
        dim_hidden = self.dim_heads * heads

        self.heads = heads
        self.to_q = nn.Linear(inchannel, dim_hidden, bias=False)
        self.to_kv = nn.Linear(inchannel, 2 * dim_hidden, bias=False)
        self.to_out = nn.Linear(dim_hidden, inchannel)

    def forward(self, x, kv=None):
        kv = x if kv is None else kv
        q, k, v = (self.to_q(x), *self.to_kv(kv).chunk(2, dim=-1))

        b, t, d, h, e = *q.shape, self.heads, self.dim_heads

        merge_heads = lambda x: x.reshape(b, -1, h, e).transpose(1, 2).reshape(b * h, -1, e)
        q, k, v = map(merge_heads, (q, k, v))

        dots = torch.einsum('bie,bje->bij', q, k) * (e ** -0.5)
        dots = dots.softmax(dim=-1)
        out = torch.einsum('bij,bje->bie', dots, v)

        out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        out = self.to_out(out)
        return out


class AxialAttention(nn.Module):
    def __init__(self, size, dims=3, heads=8, dim_heads=None, dim_index=-1, sum_axial_out=True, use_checkpoint=True):
        assert (size % heads) == 0, 'hidden dimension must be divisible by number of heads'
        super().__init__()
        self.size = size
        self.total_dimensions = dims + 2
        self.dim_index = dim_index if dim_index > 0 else (dim_index + self.total_dimensions)
        self.use_checkpoint = use_checkpoint

        attentions = []
        for permutation in calculate_permutations(dims, dim_index):
            attentions.append(PermuteToFrom(permutation, SelfAttention_Axial(size, heads, dim_heads)))

        self.axial_attentions = nn.ModuleList(attentions)
        self.sum_axial_out = sum_axial_out

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(),
                          self.use_checkpoint)  # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        # return pt_checkpoint(self._forward, x)  # pytorch

    def _forward(self, x):
        assert len(x.shape) == self.total_dimensions, 'input tensor does not have the correct number of dimensions'
        assert x.shape[self.dim_index] == self.size, 'input tensor does not have the correct input dimension'

        if self.sum_axial_out:
            '''
            map 函数：对 self.axial_attentions 中的每个 axial_attn，执行 axial_attn(x)，得到一个迭代器，包含所有注意力模块的输出。
            lambda 函数：这个匿名函数接收每个 axial_attn 并将输入 x 传递给它，实现对输入的注意力计算。
            sum 函数：将 map 生成的所有输出进行相加，得到一个总的输出张量。
            '''
            return sum(map(lambda axial_attn: axial_attn(x), self.axial_attentions))

        out = x
        for axial_attn in self.axial_attentions:
            out = axial_attn(out)
        return out


# Cross Atten
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, num_heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * num_heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.num_heads = num_heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)     
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)   

        

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, ):
        
        h = self.num_heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)  # nn.Linear(context_dim, inner_dim, bias=False) (b,num,context_dim)--- (num, context_dim)* (context_dim,inner_dim) --->(b,num,inner_dim)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        # attention, what we cannot get enough of

        attn = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class BasicCrossBlock(nn.Module):
    def __init__(self, query_dim, num_heads, dim_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=query_dim, num_heads=num_heads, dim_head=dim_head,
                                    dropout=dropout)  # is a self-attention
        self.ff = FeedForward(query_dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=query_dim, context_dim=context_dim, num_heads=num_heads,
                                    dim_head=dim_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.norm3 = nn.LayerNorm(query_dim)
        self.checkpoint = checkpoint


    def forward(self, x, context=None, ):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class CrossBlock(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(self, dims, in_channels, norm=True, use_checkpoint=True, num_heads=4, dim_head=32,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.dims = dims
        self.in_channels = in_channels
        inner_dim = num_heads * dim_head
        self.norm = Normalize(in_channels)
        self.use_checkpoint = use_checkpoint

        self.proj_in = conv_nd(dims, in_channels,
                               inner_dim,
                               kernel_size=1,
                               stride=1,
                               padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicCrossBlock(inner_dim, num_heads, dim_head, dropout=dropout, context_dim=context_dim)
             for d in range(depth)]
        )


        self.proj_out = zero_module(conv_nd(dims, inner_dim,
                                            in_channels,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0))

    def forward(self, x, context=None, ):
        
        assert x is not None
#         assert context is not None
        return checkpoint(self._forward, (x, context), self.parameters(), self.use_checkpoint) 



    def _forward(self, x, context=None, ):

        if self.dims == 3:
            b, c, h, w, d = x.shape
            x_in = x

            # Normalization
            x = self.norm(x)

            # Projection to inner dimension
            x = self.proj_in(x)

            # Reshape: from (b, c, h, w, d) -> (b, t, c), where t = h * w * d
            x = rearrange(x, 'b c h w d -> b (h w d) c')

            # Transformer blocks
            for block in self.transformer_blocks:
                x = block(x, context=context)

            x = rearrange(x, 'b (h w d) c -> b c h w d', h=h, w=w, d=d)

        if self.dims == 2:
            b, c, h, w = x.shape
            x_in = x

            # Normalization
            x = self.norm(x)

            # Projection to inner dimension
            x = self.proj_in(x)

            # Reshape: from (b, c, h, w, d) -> (b, t, c), where t = h * w * d
            x = rearrange(x, 'b c h w -> b (h w) c')

            # Transformer blocks
            for block in self.transformer_blocks:
                x = block(x, context=context)

            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

            # Output projection
        x = self.proj_out(x)

        # Skip connection
        return x + x_in


class PositionEmbedding3D(nn.Module):

    def __init__(self, depth, dim):
        super().__init__()
        self.depth = depth
        self.pos_embed = nn.Parameter(torch.randn(depth, dim))  # (depth, dim)

    def forward(self, idx):
        # idx: 切片在3D中的位置索引（如0~63）
        pos = self.pos_embed[idx].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1, dim, 1, 1)
        return pos


class Simple_Cross(nn.Module):

    def __init__(self, dims, inchannel=64, num_heads=8, bias=False, depth=96, use_checkpoint=True,**kwargs):
        super().__init__()
        self.dims = dims
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))


        self.pos_encoding = PositionEmbedding3D(depth, inchannel)
        self.q_dwconv = conv_nd(dims,inchannel, inchannel, kernel_size=3, padding=1, groups=inchannel, bias=bias)
        self.kv = conv_nd(dims,1, inchannel * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = conv_nd(dims,inchannel * 2, inchannel * 2, kernel_size=3, padding=1, groups=inchannel * 2,bias=bias)
        self.project_out = conv_nd(dims,inchannel, inchannel, kernel_size=1, bias=bias)

    def _forward(self, x_3d, y_2d, slice_idx):
        b, c, h, w, d = x_3d.shape

        x_2d = rearrange(x_3d, 'b c h w d -> b (c d) h w')
        q = self.q_dwconv(self.q(x_2d))
        pos = self.pos_encoding(slice_idx).expand(b, -1, h, w)
        y_2d_expanded = y_2d.repeat(1, d, 1, 1)
        kv = self.kv_dwconv(self.kv(y_2d_expanded + pos))  # 注入位置信息

        k, v = kv.chunk(2, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        out = rearrange(out, 'b (c d) h w -> b c h w d', c=c, d=d)
        return out

    def forward(self, x_3d, y_2d, slice_idx):
        return checkpoint(self._forward, (x_3d, y_2d, slice_idx,), self.parameters(),
                          self.use_checkpoint)





# Linear Atten
class LinearAttention(nn.Module):
    def __init__(self, dims, in_channels, norm=True, use_checkpoint=True, num_heads=8, dim_head=32, **kwargs):
        super().__init__()
        self.dims = dims
        self.num_heads = num_heads
        hidden_dim = dim_head * num_heads
        self.to_qkv = conv_nd(dims, in_channels, hidden_dim * 3, kernel_size=1, bias=False)
        self.to_out = conv_nd(dims, hidden_dim, in_channels, kernel_size=1)
        self.use_checkpoint = use_checkpoint

        if norm:
            self.norm = Normalize(in_channels)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(),
                          self.use_checkpoint) 
    def _forward(self, x):
        if self.dims == 3:
            b, c, d, h, w = x.shape
        else:
            b, c, d, h = x.shape

        h_ = x
        x = self.norm(x)

        qkv = self.to_qkv(x)  # 计算 Q, K, V


        if self.dims == 3:
            q, k, v = rearrange(qkv, 'b (qkv heads c) d h w -> qkv b heads c (d h w)', heads=self.num_heads, qkv=3)
        else:
            q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=self.num_heads, qkv=3)

        k = k.softmax(dim=-1) 
        context = torch.einsum('bhdn,bhen->bhde', k, v) 

        out = torch.einsum('bhde,bhdn->bhen', context, q) 

        if self.dims == 3:
            out = rearrange(out, 'b heads c (d h w) -> b (heads c) d h w', heads=self.num_heads, d=d, h=h, w=w) 
        else:
            out = rearrange(out, 'b heads c (d h) -> b (heads c) d h', heads=self.num_heads, d=d, h=h)  

        assert torch.isfinite(out).all(), "out output contains NaN or inf"
        return self.to_out(out) + h_ 



# Self Atten
class AttnBlock(nn.Module):
    def __init__(self, dims, in_channels, norm=True, use_checkpoint=True, num_heads=-1, dim_head=-1, **kwargs):
        super().__init__()
        self.dims = dims
        self.in_channels = in_channels
        self.use_checkpoint = use_checkpoint

        self.norm = Normalize(in_channels)
        self.q = conv_nd(dims, in_channels, in_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)
        self.k = conv_nd(dims, in_channels,
                         in_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)
        self.v = conv_nd(dims, in_channels,
                         in_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)
        self.proj_out = conv_nd(dims, in_channels,
                                in_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0)

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(),
                          self.use_checkpoint) 

    def _forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        if self.dims == 3:
            b, c, d, h, w = q.shape
            q = q.reshape(b, c, d * h * w)
            q = q.permute(0, 2, 1)  # b,hwd,c
            k = k.reshape(b, c, d * h * w)  # b,c,hwd
            w_ = torch.bmm(q, k)  # b,hwd,hwd   w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
            w_ = w_ * (int(c) ** (-0.5))
            w_ = torch.nn.functional.softmax(w_, dim=2)

            # attend to values
            v = v.reshape(b, c, d * h * w)
            w_ = w_.permute(0, 2, 1)  # b,hw,hwd (first hw of k, second of q)
            h_ = torch.bmm(v, w_)  # b, c,hwd (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
            h_ = h_.reshape(b, c, d, h, w)
        else:
            b, c, d, h = q.shape
            q = q.reshape(b, c, d * h)
            q = q.permute(0, 2, 1)  # b,hw,c
            k = k.reshape(b, c, d * h)  # b,c,hw
            w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
            w_ = w_ * (int(c) ** (-0.5))
            w_ = torch.nn.functional.softmax(w_, dim=2)

            # attend to values
            v = v.reshape(b, c, d * h)
            w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
            h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
            h_ = h_.reshape(b, c, d, h)

        h_ = self.proj_out(h_)

        return x + h_


# qkv mine for Self AttentionBlock 1D sequence pixel token with multi heand
class QKVAttentionLegacy(nn.Module):

    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.

        """
        bs, width, length = qkv.shape

        assert width % (3 * self.num_heads) == 0
        ch = width // (3 * self.num_heads)
        q, k, v = qkv.reshape(bs * self.num_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.torch.einsum("bts,bcs->bct", weight, v)
        a = a.reshape(bs, -1, length)

        assert torch.isfinite(a).all(), "a contains NaN or inf"
        return a

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.num_heads) == 0
        ch = width // (3 * self.num_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.num_heads, ch, length),
            (k * scale).view(bs * self.num_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.num_heads, ch, length))
        a = a.reshape(bs, -1, length)

        assert torch.isfinite(a).all(), "a contains NaN or inf"
        return a

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, dims=1, in_channels=None, norm=True, use_checkpoint=True, num_heads=1, dim_head=-1,
                 use_new_attention_order=False,
                 ):
        super().__init__()
        self.in_channels = in_channels

        # head
        if dim_head == -1:
            self.num_heads = num_heads
        else:
            assert (
                    in_channels % dim_head == 0
            ), f"q,k,v channels {in_channels} is not divisible by dim_head {dim_head}"
            self.num_heads = in_channels // dim_head

        self.use_checkpoint = use_checkpoint
        self.norm = Normalize(in_channels)
        self.qkv = conv_nd(1, in_channels, in_channels * 3, 1)
        if use_new_attention_order:  # False
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, in_channels, in_channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(),
                          self.use_checkpoint)  # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        # return pt_checkpoint(self._forward, x)  # pytorch

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)

        return (x + h).reshape(b, c, *spatial)

    
    
    
    
    
class FreeCrossAttention(nn.Module):
    '''
    {
      "text_label_mapping": {
        "1": 0.53,
        "2": 0.47,
        "3": 0.66,
      },
      "layout_path": "examples/layout_boat.png"
    }
    index_ = [0.53,0.47,0.66]
    index = [1, 2, 3]
    label = (h,w,d)
    '''    
    def __init__(self, query_dim, context_dim=None, num_heads=8, dim_head=64, dropout=0.):
        super().__init__()
        
        self.max_slice = 32
        
        inner_dim = dim_head * num_heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = num_heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, label=None, class_ids=None):
        
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        if context.shape[-2]>5:
            mask_size = self.max_slice
        else:
            mask_size = -1

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))


        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(label) and exists(class_ids):
            B = label.shape[0]
            H= int(q.shape[1] ** (1 / 3) + 1e-5)
            W = H
            D = H

            mask = torch.ones(B, H, W, D, context.shape[-2])   # context.shape = (b,num_area,context_dim)

            for ii in range(B):
                index = class_ids[ii]
                if mask_size >0:
                    index = index.repeat_interleave(mask_size)

                for kk in range(len(index)):
                    if index[kk] == -1:
                        continue
                    tmp_mask = torch.zeros_like(label[ii]) # [h,w,d]
                    tmp_mask[label[ii]==index[kk]] = 1
                    tmp_mask = tmp_mask.float()
                    tmp_mask = F.interpolate(tmp_mask.unsqueeze(0).unsqueeze(0), (H, W, D), mode="nearest")[0,0,:,:]
                    mask[ii,:,:,:,kk] = tmp_mask
                    del tmp_mask
                                               
            mask = rearrange(mask, 'b h w d c-> b (h w d) c')
            mask = repeat(mask, 'b n c-> (b h) n c', h=h)
            mask = mask.to(q.device)
            mask = mask > 0.5
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg_value)
            del mask

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)



class FreeCrossBlock(nn.Module):
    def __init__(self, query_dim, num_heads, dim_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = FreeCrossAttention(query_dim=query_dim, num_heads=num_heads, dim_head=dim_head,
                                    dropout=dropout) 
        self.ff = FeedForward(query_dim, dropout=dropout, glu=gated_ff)
        self.attn2 = FreeCrossAttention(query_dim=query_dim, context_dim=context_dim, num_heads=num_heads,
                                    dim_head=dim_head, dropout=dropout)  
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.norm3 = nn.LayerNorm(query_dim)
        self.checkpoint = checkpoint


    def forward(self, x, context=None, label=None, class_ids=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context, label=label, class_ids=class_ids) + x
        x = self.ff(self.norm3(x)) + x
        return x


class FreeCrossModule(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(self, dims, in_channels, norm=True, use_checkpoint=True, num_heads=4, dim_head=32,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.dims = dims
        self.in_channels = in_channels
        inner_dim = num_heads * dim_head
        self.norm = Normalize(in_channels)
        self.use_checkpoint = use_checkpoint

        self.proj_in = conv_nd(dims, in_channels,
                               inner_dim,
                               kernel_size=1,
                               stride=1,
                               padding=0)

        self.transformer_blocks = nn.ModuleList(
            [FreeCrossBlock(inner_dim, num_heads, dim_head, dropout=dropout, context_dim=context_dim)
             for d in range(depth)]
        )


        self.proj_out = zero_module(conv_nd(dims, inner_dim,
                                            in_channels,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0))

    def forward(self, x, context=None,label=None, class_ids=None):
        
        assert x is not None
        assert context is not None
        return checkpoint(self._forward, (x, context, label, class_ids), self.parameters(), self.use_checkpoint) 



    def _forward(self, x, context=None,label=None, class_ids=None ):

        if self.dims == 3:
            b, c, h, w, d = x.shape
            x_in = x

            x = self.norm(x)

            x = self.proj_in(x)

            x = rearrange(x, 'b c h w d -> b (h w d) c')

            for block in self.transformer_blocks:
                x = block(x, context=context, label=label, class_ids=class_ids)

            x = rearrange(x, 'b (h w d) c -> b c h w d', h=h, w=w, d=d)

        if self.dims == 2:
            b, c, h, w = x.shape
            x_in = x

            x = self.norm(x)

            x = self.proj_in(x)

            x = rearrange(x, 'b c h w -> b (h w) c')

            for block in self.transformer_blocks:
                x = block(x, context=context)

            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        x = self.proj_out(x)

        return x + x_in
    
    
    
    
    

#
def make_attn(dims, in_channels, use_checkpoint, norm=True, num_heads=8, dim_head=-1, attn_type="basic", context_dim=None, **kwargs):
    assert attn_type in ["basic", "basic_", "linear", "linear_n", "cross", "axial", "sim", "free_cross",
                         "none"], f'attn_type {attn_type} unknown'

    if attn_type == "basic":
        return AttnBlock(dims=dims, in_channels=in_channels, use_checkpoint=use_checkpoint, norm=norm,
                         num_heads=num_heads, dim_head=dim_head, **kwargs)

    if attn_type == "basic_":
        return AttentionBlock(dims=1, in_channels=in_channels, use_checkpoint=use_checkpoint, norm=norm,
                              num_heads=num_heads, dim_head=dim_head, **kwargs)

    elif attn_type == "linear":
        return LinearAttention(dims=dims, in_channels=in_channels, use_checkpoint=use_checkpoint, norm=norm,
                               num_heads=num_heads, dim_head=dim_head, **kwargs)

    elif attn_type == "cross":
#         assert context_dim is not None
        return CrossBlock(dims=dims, in_channels=in_channels, use_checkpoint=use_checkpoint, norm=True,
                          num_heads=num_heads, dim_head=dim_head, context_dim=context_dim, **kwargs)
    
    elif attn_type == "free_cross":
#         assert context_dim is not None
        return FreeCrossModule(dims=dims, in_channels=in_channels, use_checkpoint=use_checkpoint, norm=True,
                          num_heads=num_heads, dim_head=dim_head, context_dim=context_dim, **kwargs)


    elif attn_type == "simple_cross":
        return Simple_Cross(dims=dims, in_channels=in_channels, use_checkpoint=use_checkpoint, norm=norm,
                          num_heads=num_heads, dim_head=dim_head, **kwargs)

    elif attn_type == "sim":
        return SimAM(dims=dims, in_channels=in_channels, use_checkpoint=use_checkpoint, norm=norm, num_heads=num_heads,
                     dim_head=dim_head, **kwargs)

    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return None



def run_2d():
    batch_size = 1
    in_channels = 64
    height = 32
    width = 32
    context_dim = 768
    num_heads = 4
    dim_head = 32
    dims = 2

    x = torch.randn(batch_size, in_channels, height, width)
    context = torch.randn(batch_size, 77, context_dim)  # 假设序列长度为77

    model = CrossBlock(
        dims=dims,
        in_channels=in_channels,
        num_heads=num_heads,
        dim_head=dim_head,
        context_dim=context_dim
    )

    output = model(x, context=context)

    assert output.shape == (batch_size, in_channels, height, width), f"Expected shape {(batch_size, in_channels, height, width)}, got {output.shape}"
    print("2D Test Passed")

def run_3d():

    batch_size = 1
    in_channels = 64
    depth = 8
    height = 32
    width = 32
    context_dim = 768
    num_heads = 4
    dim_head = 32
    dims = 3

    x = torch.randn(batch_size, in_channels, depth, height, width)
    context = torch.randn(batch_size, 77, context_dim)  # 假设序列长度为77

    model = CrossBlock(
        dims=dims,
        in_channels=in_channels,
        num_heads=num_heads,
        dim_head=dim_head,
        context_dim=context_dim
    )

    output = model(x, context=context)

    assert output.shape == (batch_size, in_channels, depth, height, width), f"Expected shape {(batch_size, in_channels, depth, height, width)}, got {output.shape}"
    print("3D Test Passed")

if __name__ == "__main__":
    run_2d()
    run_3d()
