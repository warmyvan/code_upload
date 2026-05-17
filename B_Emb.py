

from F_fct_module import *


class SF_emb(nn.Module):
    def __init__(self, base_chnl, embeding_dim, out_chnl, num_classes, modes=None, emb_Frq=False, trnbl=False):
        super().__init__()
        
        self.embeding_dim = embeding_dim
        self.num_classes = num_classes
        self.base_chnl = base_chnl
        self.modes = modes
        self.out_chnl = out_chnl        
        self.emb_Frq = emb_Frq
        self.trnbl = trnbl

        # t_emb to emb
        self.time_embed = nn.Sequential(
            nn.Linear(base_chnl, embeding_dim),
            nn.SiLU(),
            nn.Linear(embeding_dim, embeding_dim),
        )
        # class emb
        if self.num_classes:
            print('LABELLLLLL!!!')
            self.emb_y = nn.Embedding(num_classes, embeding_dim)
            
        # F_emb
        if self.emb_Frq:
            self.emb_frqc_real = nn.Sequential(
                nn.SiLU(),
                linear(embeding_dim,out_chnl),
            )
            self.emb_frqc_imag = nn.Sequential(
                nn.SiLU(),
                linear(embeding_dim,out_chnl),
            )
    def forward(self, x, timesteps=None, y=None, trnbl=True, context=None, label=None, class_ids=None,**kwargs):

        # time
        t_emb = timestep_embedding(timesteps, self.base_chnl, repeat_only=False)
        emb = self.time_embed(t_emb)

        # emb_S
        if self.num_classes is not None:
            print('LABELLLLLL!!!')
            assert y.shape[0] == x.shape[0]
            emb = emb + self.emb_y(y)

        if self.emb_Frq == True:

            # emb_F1
            if trnbl ==True:
                real = self.emb_frqc_real(emb)  # (b, out_chnl)
                imag = self.emb_frqc_imag(emb)  # (b, out_chnl)

                emb_F = torch.complex(real, imag)  

                assert torch.isfinite(emb_F).all(), "emb contains NaN or inf"
                return emb, emb_F


        assert torch.isfinite(emb).all(), "emb contains NaN or inf"
        emb_F = None
        return emb, emb_F  # (b,embedding_dim)


def SF_emb_run():

    batch_size = 2
    embedding_dim = 125
    base_chnl = 64
    out_chnl = 64
    modes = 4
    num_cls = 5

    model = SF_emb(embedding_dim, base_chnl, modes, out_chnl, num_cls)

    timesteps = torch.rand(batch_size)
    y = torch.randint(0, num_cls, (batch_size,)) 
    x_dummy = torch.randn(batch_size, 3, 32, 32, 32) 

    trainable_complex = model(x_dummy, timesteps=timesteps, y=y,trnbl=True)
    assert trainable_complex.shape == (batch_size, out_chnl,), "shape error"
    assert torch.is_complex(trainable_complex), "complex tensor"


    assert torch.isfinite(trainable_complex.real).all() and torch.isfinite(trainable_complex.imag).all()

if __name__ == "__main__":
    SF_emb_run()

