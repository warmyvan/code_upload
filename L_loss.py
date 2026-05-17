from  L_kl_g_loss import *
from L_ploss_vgg16_2d import *
import torch.nn as nn


def adopt_weight(disc_factor, global_step, threshold=0, value=0.):
    if global_step < threshold:
        disc_factor = value
    return disc_factor


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss



def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, *, logvar_init=0.0,
                 kl_weight=0.00001,
                 perceptual_weight=0.0,
                 disc_start,
                 disc_num_layers=3,
                 disc_in_channels=3,
                 disc_factor=0.0,
                 disc_weight=0.0,
                 disc_conditional=False,
                 disc_loss="hinge",
                 use_actnorm=False,
                 **ignorekwargs):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla", None]

        # raw_loss weight
        self.kl_weight = kl_weight
        self.perceptual_loss = nn.Identity()

        self.perceptual_weight = perceptual_weight

        self.logvar = torch.tensor(logvar_init)

        # about disc
        self.discriminator = NLayerDiscriminator3D(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init) if disc_loss is not None else nn.Identity
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        # assert 0==1

    def calculate_adaptive_d_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs,
                reconstructions,
                posteriors,
                loss_choice,
                weights=None,
                global_step=None,
                last_layer=None,
                cond=None,
                split='E',
                seg=0.5
                ):

        loss_fn = nn.L1Loss(reduction='none')
        rec_loss = loss_fn(inputs, reconstructions).mean(dim=(1, 2, 3, 4))  # per-sample loss
        
        # perceptual raw_loss --> rec_loss
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        # rec_loss --> nll raw_loss
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar


        # kl_loss #

        kl_loss = posteriors.kl()


        # WITH disc raw_loss
        if loss_choice == 0:
            # g_loss
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))

            g_loss = -torch.mean(logits_fake)

            # d_weight
            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_d_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            # disc_factor
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)

            loss = rec_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss


            # log total_loss/rec_loss/kl_loss/g_loss #
            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): disc_factor.detach(),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        # ONLY disc raw_loss
        if loss_choice == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)

            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }

            return d_loss, log

        # NO disc raw_loss
        if loss_choice == 2:
            g_loss = torch.tensor(0.0)

            d_weight = torch.tensor(0.0)

            disc_factor = torch.tensor(0.0)

            # raw_loss == total_loss == weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss #
            loss = rec_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss
            print('\n')
            print(f'nll_loss:{nll_loss}')
            print(f'self.kl_weight * kl_los={self.kl_weight}*{kl_loss}')
            print(f'loss:{loss}')

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   }
            return loss, log
