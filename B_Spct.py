from F_fct_module import *

# fft #
class SpectralConv3d(nn.Module):

    def __init__(self, in_channels, out_channels, size, max_modes, use_checkpoint=False, **kwargs):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_ = size[2]//2 if size[2]//2 <= max_modes else max_modes
        self.use_checkpoint = use_checkpoint

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes_, self.modes_, self.modes_, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes_, self.modes_, self.modes_, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes_, self.modes_, self.modes_, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes_, self.modes_, self.modes_, dtype=torch.cfloat))

    def compl_mul3d(self, matrix_A, matrix_B):
        return torch.einsum("bixyz,ioxyz->boxyz", matrix_A, matrix_B)


    def forward(self, x, emb_F=None, **kwargs):
        return checkpoint(
            self._forward, (x, emb_F), self.parameters(), self.use_checkpoint
        )


    def forward_(self, x, emb_F=None, **kwargs):

        batchsize = x.shape[0]

        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1], norm="ortho")

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                          device=x.device)
        out_ft[:, :, :self.modes_, :self.modes_, :self.modes_] = \
            self.compl_mul3d(x_ft[:, :, :self.modes_, :self.modes_, :self.modes_], self.weights1)

        out_ft[:, :, -self.modes_:, :self.modes_, :self.modes_] = \
            self.compl_mul3d(x_ft[:, :, -self.modes_:, :self.modes_, :self.modes_], self.weights2)

        out_ft[:, :, :self.modes_, -self.modes_:, :self.modes_] = \
            self.compl_mul3d(x_ft[:, :, :self.modes_, -self.modes_:, :self.modes_], self.weights3)

        out_ft[:, :, -self.modes_:, -self.modes_:, :self.modes_] = \
            self.compl_mul3d(x_ft[:, :, -self.modes_:, -self.modes_:, :self.modes_], self.weights4)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)), norm="ortho")
        return x



class SpectralConvnd_SR(nn.Module):
    def __init__(self, in_channels, dims=3, use_checkpoint=False, **kwargs):
        super(SpectralConvnd_SR, self).__init__()

        self.dims = dims
        self.use_checkpoint = use_checkpoint

        self.fpre = conv_nd(dims, in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.amp_fuse = nn.Sequential(conv_nd(dims, in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                                      nn.LeakyReLU(0.1, inplace=True),
                                      conv_nd(dims, in_channels, in_channels, kernel_size=3, stride=1, padding=1))
        self.pha_fuse = nn.Sequential(conv_nd(dims, in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                                      nn.LeakyReLU(0.1, inplace=True),
                                      conv_nd(dims, in_channels, in_channels, kernel_size=3, stride=1, padding=1))
        self.post = conv_nd(dims, in_channels, in_channels, kernel_size=1, stride=1, padding=0)



    def forward(self, x, emb_F=None, **kwargs):
        return checkpoint(
            self.forward_, (x, emb_F), self.parameters(), self.use_checkpoint
        )


    def forward_(self, x, emb_F=None, **kwargs):
        if self.dims ==3:
            _, _, H, W, D = x.shape
        else:
            _, _, H, W = x.shape

        msF = torch.fft.rfftn(self.fpre(x) + 1e-8, dim=[-3, -2, -1], norm='backward') if self.dims==3 else torch.fft.rfftn(self.fpre(x) + 1e-8, dim=[-2, -1], norm='backward')


        msF_amp = torch.abs(msF)
        msF_pha = torch.angle(msF)

        amp_fuse = self.amp_fuse(msF_amp)
        amp_fuse = amp_fuse + msF_amp

        pha_fuse = self.pha_fuse(msF_pha)
        pha_fuse = pha_fuse + msF_pha

        real = amp_fuse * torch.cos(pha_fuse) + 1e-8
        imag = amp_fuse * torch.sin(pha_fuse) + 1e-8
        out = torch.complex(real, imag) + 1e-8

        out = torch.abs(torch.fft.irfftn(out, s=(H, W, D), norm="backward")) if self.dims==3 else torch.abs(torch.fft.irfftn(out, s=(H, W), norm="backward"))

        out = self.post(out)

        out = out + x
        out_ = torch.nan_to_num(out, nan=1e-5, posinf=1e-5, neginf=1e-5)
        return out_





class SpectralConv3d_Grid(nn.Module):

    def __init__(self, in_channels, out_channels, size, max_modes, use_checkpoint=False, **kwargs):
        super(SpectralConv3d_Grid, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_ = size[2]//2 if size[2]//2 <= max_modes else max_modes
        self.use_checkpoint = use_checkpoint

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes_, self.modes_, self.modes_, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes_, self.modes_, self.modes_, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes_, self.modes_, self.modes_, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes_, self.modes_, self.modes_, dtype=torch.cfloat))


    def compl_mul3d(self, matrix_A, matrix_B):
        return torch.einsum("bixyz,ioxyz->boxyz", matrix_A, matrix_B)


    def get_grid(self, batchsize, size_x, size_y, size_z, device):
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1, 1).repeat([batchsize, size_x, size_y, 1, 1])

        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)  #


    def forward(self, x, emb_F=None, **kwargs):
        return checkpoint(
            self._forward, (x, emb_F), self.parameters(), self.use_checkpoint
        )


    def forward_(self, x, emb_F=None, **kwargs):
        '''
        (2,1,96,96,96)-- (2,widtorch,96,96,97)--  (2,width,96,96,96)
        '''
        batchsize = x.shape[0]
        grid = self.get_grid(batchsize, x.shape[1], x.shape[2], x.shape[3], x.shape[4], x.device)
        x = torch.cat((x, grid), dim=-1)

        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1], norm="ortho")

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                          device=x.device)
        out_ft[:, :, :self.modes_, :self.modes_, :self.modes_] = \
            self.compl_mul3d(x_ft[:, :, :self.modes_, :self.modes_, :self.modes_], self.weights1)

        out_ft[:, :, -self.modes_:, :self.modes_, :self.modes_] = \
            self.compl_mul3d(x_ft[:, :, -self.modes_:, :self.modes_, :self.modes_], self.weights2)

        out_ft[:, :, :self.modes_, -self.modes_:, :self.modes_] = \
            self.compl_mul3d(x_ft[:, :, :self.modes_, -self.modes_:, :self.modes_], self.weights3)

        out_ft[:, :, -self.modes_:, -self.modes_:, :self.modes_] = \
            self.compl_mul3d(x_ft[:, :, -self.modes_:, -self.modes_:, :self.modes_], self.weights4)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)), norm="ortho")
        return x



# fft embed #
class Embed_SpectralConv3d(nn.Module):

    def __init__(self, in_channels, out_channels, size, max_modes, use_checkpoint=False, **kwargs):
        super(Embed_SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_ = size[2]//2 if size[2]//2 <= max_modes else max_modes
        self.use_checkpoint = use_checkpoint


        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes_, self.modes_, self.modes_, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes_, self.modes_, self.modes_, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes_, self.modes_, self.modes_, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes_, self.modes_, self.modes_, dtype=torch.cfloat))

    def compl_mul3d(self, matrix_A, matrix_B):
        return torch.einsum("bixyz,ioxyz->boxyz", matrix_A, matrix_B)


    def forward(self, x, emb_F=None, **kwargs):
        return checkpoint(
            self._forward, (x, emb_F), self.parameters(), self.use_checkpoint
        )


    def forward_(self, x, emb_F=None, **kwargs):

        batchsize = x.shape[0]

        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1], norm="ortho")

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                          device=x.device)
        out_ft[:, :, :self.modes_, :self.modes_, :self.modes_] = \
            self.compl_mul3d(x_ft[:, :, :self.modes_, :self.modes_, :self.modes_], self.weights1)

        out_ft[:, :, -self.modes_:, :self.modes_, :self.modes_] = \
            self.compl_mul3d(x_ft[:, :, -self.modes_:, :self.modes_, :self.modes_], self.weights2)

        out_ft[:, :, :self.modes_, -self.modes_:, :self.modes_] = \
            self.compl_mul3d(x_ft[:, :, :self.modes_, -self.modes_:, :self.modes_], self.weights3)

        out_ft[:, :, -self.modes_:, -self.modes_:, :self.modes_] = \
            self.compl_mul3d(x_ft[:, :, -self.modes_:, -self.modes_:, :self.modes_], self.weights4)

        if emb_F is not None:
            while len(emb_F.shape) < len(out_ft.shape):
                emb_F = emb_F[..., None]
            out_ft = out_ft + emb_F

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)), norm="ortho")

        return x


    
    

# method 1.1
class Embed_SpectralConv3d_FreqSplit(nn.Module):

    def __init__(self, in_channels, out_channels, size, max_modes,
                 emb_dim, freq_split=False, highfrq_modulation=False, use_checkpoint=False, **kwargs):
        super(Embed_SpectralConv3d_FreqSplit, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_ = size[2]//2 if size[2]//2 <= max_modes else max_modes
        self.frqsplt =freq_split
        self.highfrq_modulation = highfrq_modulation
        self.use_checkpoint = use_checkpoint

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes_, self.modes_, self.modes_, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes_, self.modes_, self.modes_, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes_, self.modes_, self.modes_, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes_, self.modes_, self.modes_, dtype=torch.cfloat))

        if freq_split == True:
            assert emb_dim != 0,'emb_dim Error from Embed_SpectralConv3d_FreqSplit __init__'
            self.emb_mode = nn.Embedding(self.modes_, emb_dim)
            self.mapper = linear(emb_dim * 2, emb_dim * 4)


    def compl_mul3d(self, matrix_A, matrix_B):
        return torch.einsum("bixyz,ioxyz->boxyz", matrix_A, matrix_B)


    def forward(self, x, emb_F=None, **kwargs):
        return checkpoint(
            self._forward, (x, emb_F), self.parameters(), self.use_checkpoint
        )


    def forward_(self, x, emb_F=None, **kwargs):

        batchsize = x.shape[0]

        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1], norm="ortho")
        # x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1], norm="backward")

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                          device=x.device)
        out_ft[:, :, :self.modes_, :self.modes_, :self.modes_] = \
            self.compl_mul3d(x_ft[:, :, :self.modes_, :self.modes_, :self.modes_], self.weights1)

        out_ft[:, :, -self.modes_:, :self.modes_, :self.modes_] = \
            self.compl_mul3d(x_ft[:, :, -self.modes_:, :self.modes_, :self.modes_], self.weights2)

        out_ft[:, :, :self.modes_, -self.modes_:, :self.modes_] = \
            self.compl_mul3d(x_ft[:, :, :self.modes_, -self.modes_:, :self.modes_], self.weights3)

        out_ft[:, :, -self.modes_:, -self.modes_:, :self.modes_] = \
            self.compl_mul3d(x_ft[:, :, -self.modes_:, -self.modes_:, :self.modes_], self.weights4)

        out_ft_ = out_ft

        if emb_F is not None:
            if self.frqsplt == True:
                embedded_mode = self.emb_mode(self.modes_)
                embedded_mode = embedded_mode.expand(batchsize, -1) 

                combined = torch.cat([emb_F, embedded_mode], dim=1)  
                mapped = self.mapper(combined)  # (B, embedding_dim*4)
                split1, split2 = mapped.chunk(2, dim=1)  # (B, embedding_dim*2)

                scale_low, bias_low = split1.chunk(2, dim=1)  # (B, embedding_dim)
                scale_high, bias_high = split2.chunk(2, dim=1)

                while len(scale_low.shape) < len(out_ft.shape):
                    scale_low = scale_low[..., None]
                    bias_low = bias_low[..., None]
                    scale_high = scale_high[..., None]
                    bias_high = bias_high[..., None]

                out_ft[:, :, :self.modes_, :self.modes_, :self.modes_] = \
                    out_ft[:, :, :self.modes_, :self.modes_, :self.modes_] * (1 + scale_low) + bias_low

                out_ft[:, :, -self.modes_:, :self.modes_, :self.modes_] = \
                    out_ft[:, :, -self.modes_:, :self.modes_, :self.modes_] * (1 + scale_low) + bias_low

                out_ft[:, :, :self.modes_, -self.modes_:, :self.modes_] = \
                    out_ft[:, :, :self.modes_, -self.modes_:, :self.modes_] * (1 + scale_low) + bias_low

                out_ft[:, :, -self.modes_:, -self.modes_:, :self.modes_] = \
                    out_ft[:, :, -self.modes_:, -self.modes_:, :self.modes_] * (1 + scale_low) + bias_low


                if self.highfrq_modulation ==True:
                    out_ft_ = out_ft_ * (1 + scale_low) + bias_low
                    out_ft_[:, :, :self.modes_, :self.modes_, :self.modes_] = 0
                    out_ft_[:, :, -self.modes_:, :self.modes_, :self.modes_] = 0
                    out_ft_[:, :, :self.modes_, -self.modes_:, :self.modes_] = 0
                    out_ft_[:, :, -self.modes_:, -self.modes_:, :self.modes_] = 0

                    out_ft = out_ft + out_ft_

            else:
                while len(emb_F.shape) < len(out_ft.shape):
                    emb_F = emb_F[..., None]
                out_ft = out_ft + emb_F

        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)), norm="ortho")

        return x


    
    
# method 1.0
class Embed_SpectralConv3d_(nn.Module):

    def __init__(self, in_channels, out_channels, size, max_modes, use_checkpoint=False, dims=3,  **kwargs):
        super(Embed_SpectralConv3d_, self).__init__()
        self.dims =dims
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_ = size[2]//2 if size[2]//2 <= max_modes else max_modes
        self.use_checkpoint = use_checkpoint


        self.scale = (1 / (in_channels * out_channels))
        if dims==3:
            self.weights1 = nn.Parameter(
                self.scale * torch.rand(in_channels, out_channels, self.modes_, self.modes_, self.modes_, dtype=torch.cfloat))
            self.weights2 = nn.Parameter(
                self.scale * torch.rand(in_channels, out_channels, self.modes_, self.modes_, self.modes_, dtype=torch.cfloat))
            self.weights3 = nn.Parameter(
                self.scale * torch.rand(in_channels, out_channels, self.modes_, self.modes_, self.modes_, dtype=torch.cfloat))
            self.weights4 = nn.Parameter(
                self.scale * torch.rand(in_channels, out_channels, self.modes_, self.modes_, self.modes_, dtype=torch.cfloat))

        elif dims == 2:
            self.weights1 = nn.Parameter(
                self.scale * torch.rand(in_channels, out_channels, self.modes_, self.modes_, dtype=torch.cfloat))
            self.weights2 = nn.Parameter(
                self.scale * torch.rand(in_channels, out_channels, self.modes_, self.modes_, dtype=torch.cfloat))

    def compl_mul3d(self, matrix_A, matrix_B):
        return torch.einsum("bixyz,ioxyz->boxyz", matrix_A, matrix_B)
        
    def compl_mul2d(self, matrix_A, matrix_B):
        return torch.einsum("bixy,ioxy->boxy", matrix_A, matrix_B)


    def forward(self, x, emb_F=None, **kwargs):
        return checkpoint(
            self._forward, (x, emb_F), self.parameters(), self.use_checkpoint
        )


    def forward_(self, x, emb_F=None, **kwargs):

        batchsize = x.shape[0]
        if self.dims == 3:
            x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1], norm="ortho")

            out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                              device=x.device)
            out_ft[:, :, :self.modes_, :self.modes_, :self.modes_] = \
                self.compl_mul3d(x_ft[:, :, :self.modes_, :self.modes_, :self.modes_], self.weights1)

            out_ft[:, :, -self.modes_:, :self.modes_, :self.modes_] = \
                self.compl_mul3d(x_ft[:, :, -self.modes_:, :self.modes_, :self.modes_], self.weights2)

            out_ft[:, :, :self.modes_, -self.modes_:, :self.modes_] = \
                self.compl_mul3d(x_ft[:, :, :self.modes_, -self.modes_:, :self.modes_], self.weights3)

            out_ft[:, :, -self.modes_:, -self.modes_:, :self.modes_] = \
                self.compl_mul3d(x_ft[:, :, -self.modes_:, -self.modes_:, :self.modes_], self.weights4)
        elif self.dims == 2:
            x_ft = torch.fft.rfft2(x)
            
            out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
            out_ft[:, :, :self.modes1, :self.modes2] = \
                self.compl_mul2d(x_ft[:, :, :self.modes_, :self.modes_], self.weights1)
            out_ft[:, :, -self.modes1:, :self.modes2] = \
                self.compl_mul2d(x_ft[:, :, -self.modes_:, :self.modes_], self.weights2)

        if emb_F is not None:
            while len(emb_F.shape) < len(out_ft.shape):
                emb_F = emb_F[..., None]
            out_ft = out_ft + emb_F

        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)), norm="ortho")

        return x
    
    
    
    

# method 2
class Embed_SpectralConvnd_SR(nn.Module):

    def __init__(self, in_channels, dims=3, use_checkpoint=False, **kwargs):
        super(Embed_SpectralConvnd_SR, self).__init__()

        self.use_checkpoint = use_checkpoint

        self.dims = dims


        self.fpre = conv_nd(dims, in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.amp_fuse = nn.Sequential(conv_nd(dims, in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                                      nn.LeakyReLU(0.1, inplace=True),
                                      conv_nd(dims, in_channels, in_channels, kernel_size=3, stride=1, padding=1))
        self.pha_fuse = nn.Sequential(conv_nd(dims, in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                                      nn.LeakyReLU(0.1, inplace=True),
                                      conv_nd(dims, in_channels, in_channels, kernel_size=3, stride=1, padding=1))
        self.post = conv_nd(dims, in_channels, in_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x, emb_F=None, **kwargs):
        return checkpoint(
            self._forward, (x, emb_F), self.parameters(), self.use_checkpoint
        )


    def _forward(self, x, emb_F=None, **kwargs):

        if self.dims == 3:
            _, _, H, W, D = x.shape
            msF = torch.fft.rfftn(self.fpre(x) + 1e-8, dim=[-3, -2, -1], norm='backward')
        else:
            _, _, H, W = x.shape
            msF = torch.fft.rfftn(self.fpre(x) + 1e-8, dim=[-2, -1], norm='backward')


        msF_amp = torch.abs(msF)
        msF_pha = torch.angle(msF)

        amp_fuse = self.amp_fuse(msF_amp)
        amp_fuse = amp_fuse + msF_amp

        pha_fuse = self.pha_fuse(msF_pha)
        pha_fuse = pha_fuse + msF_pha

        real = amp_fuse * torch.cos(pha_fuse) + 1e-8
        imag = amp_fuse * torch.sin(pha_fuse) + 1e-8


        if emb_F is not None:

            out = torch.complex(real, imag) + 1e-8
            while len(emb_F.shape) < len(out.shape):
                emb_F = emb_F[..., None]
            out = out + emb_F


        else:
            out = torch.complex(real, imag) + 1e-8

        out = torch.abs(torch.fft.irfftn(out, s=(H, W, D), norm="backward")) if self.dims == 3 else torch.abs(
            torch.fft.irfftn(out, s=(H, W), norm="backward"))

        out = self.post(out)

        out = out + x
        out_ = torch.nan_to_num(out, nan=1e-5, posinf=1e-5, neginf=1e-5)

        return out_




# freq block #
class HighFreqEnhancer(nn.Module):
    def __init__(self, channels, dims=3,  use_checkpoint=False, **kwargs):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.conv = nn.Sequential(
            nn.InstanceNorm3d(channels), 
            nn.GELU(),  

            conv_nd(dims, channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(channels),  
            nn.GELU(),  
            conv_nd(dims, channels, channels, kernel_size=1),
            nn.Sigmoid()  
        )


    def _forward(self, x, emb):
        attn = self.conv(x)  
        return x * (1 + attn)  

    def forward(self, x, emb):
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )



def make_spect( in_channels, out_channels, size, max_modes, use_checkpoint, dims=3, spect_type='Emb_F3DSR',**kwargs):

    assert spect_type in ['F3D','F3DSR','Emb_F3D','Emb_F3D_splt','Emb_F3DSR'], f'spect_type {spect_type} unknown'

    if spect_type == 'F3D':
        return SpectralConv3d(in_channels, out_channels, size, max_modes, **kwargs)

    elif spect_type == 'F3DSR':
        return SpectralConvnd_SR(in_channels, dims=dims, **kwargs)

    elif spect_type == 'Emb_F3D':
        return Embed_SpectralConv3d_(in_channels, out_channels, size, max_modes, dims=dims, **kwargs)

    elif spect_type == 'Emb_F3D_splt':
        return Embed_SpectralConv3d_FreqSplit(in_channels, out_channels, size, max_modes, **kwargs)

    elif spect_type == 'Emb_F3DSR':
        return Embed_SpectralConvnd_SR(in_channels, dims=dims, **kwargs)

    else:
        return None
