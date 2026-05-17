from B_Att import *
from B_Res import *
from B_Spct import *



class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None, label=None, class_ids=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)

            elif isinstance(layer, CrossBlock):
                x = layer(x, context)
            elif isinstance(layer, FreeCrossModule):
                x = layer(x, context, label, class_ids)
            elif isinstance(layer, SpectralConv3d):
                x = layer(x)
            elif isinstance(layer, SpectralConvnd_SR):
                x = layer(x)
            elif isinstance(layer, SpectralConv3d_Grid):
                x = layer(x)
            elif isinstance(layer, Embed_SpectralConv3d):
                x = layer(x, emb)
            elif isinstance(layer, Embed_SpectralConvnd_SR):
                x = layer(x, emb)
            elif isinstance(layer, Embed_SpectralConv3d_FreqSplit):
                x = layer(x, emb)

            else:
                x = layer(x)
        return x