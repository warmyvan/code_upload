from F_fct_module import *

class PorousFusion(nn.Module):
    def __init__(self, emb_channels, mid_channels=6, dim=3):
        super().__init__()
        self.alpha_pred = nn.Sequential(
            nn.Linear(emb_channels, mid_channels),
            nn.ReLU(),
            nn.Linear(mid_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, emb, spect_out, unet_out):
        assert (spect_out.shape == unet_out.shape)

        if len(spect_out.shape)==5:
            alpha = self.alpha_pred(emb).view(-1,1,1,1,1)
        elif len(spect_out.shape)==4:
            alpha = self.alpha_pred(emb).view(-1,1,1,1)
        
        return alpha * spect_out + (1 - alpha) * unet_out ,alpha
