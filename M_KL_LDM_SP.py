# -*- coding: utf-8 -*-
"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""
from F_fct_module import *
from functools import partial
import sys
# from pytorch_lightning.utilities.distributed import rank_zero_only


sys.setrecursionlimit(10000)  # Python 默认的递归调用深度限制通常是 1000

# 由下面注释掉的 class DDPM去除不必要的参数和功能 简化而来
class DDPM_(nn.Module):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self, *,
                 sfnet,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 image_size=256,
                 channels=3,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 parameterization="eps",  # all assuming fixed variance schedules
                 learn_logvar=False,
                 logvar_init=0.,
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization  # eps 与损失计算权重有关
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.image_size = image_size
        self.channels = channels
        self.model = sfnet
        count_params(self.model, verbose=True)

        self.v_posterior = v_posterior

        self.l_simple_weight = l_simple_weight

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type

        #
        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000, linear_start=0.0015,
                          linear_end=0.01952, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            # len(betas) = ntimesteps
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and Raw
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1',
                             to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2',
                             to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))



    # # # # # noising process # # # # #
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    # # # # # denoising process # # # # #
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def p_mean_variance(self, x, t, clip_denoised: bool, context=None, label=None, class_ids=None, y=None, slice_=None):
        # model_out = self.model(x, t)
        model_out,alpha = self.model(x=x, timesteps=t, context=context, label=label, class_ids=class_ids, y=y, slice_=slice_)

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            assert 0 == 1, 'self.parameterization setting error'
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance,alpha

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False, y=None, context=None, slice_=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance,alpha = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised, y=y, context=context, slice_=slice_)
        noise = noise_like(x.shape, device, repeat_noise)

        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, alpha
    
    
    
    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False, y=None, context=None, slice_=None):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        img0 = img
        intermediates = [img]

        gate_dict = []

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img, gate = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised, context=context, slice_=slice_, y=y)
            if return_intermediates:
                if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                    intermediates.append(img)

            gate_dict.append((i,gate))

        if return_intermediates:
            return img, intermediates

        return img, img0, gate_dict

    @torch.no_grad()
    def sample(self, batch_size=4, return_intermediates=False, y=None, context=None, slice_=None, dims=2):
        image_size = self.image_size
        channels = self.channels
        if dims==3:
            return self.p_sample_loop((batch_size, channels, image_size, image_size, image_size),
                                  return_intermediates=return_intermediates, context=context, slice_=slice_,
                                  y=y)
        elif dims==2:
            return self.p_sample_loop((batch_size, channels, image_size, image_size),
                                  return_intermediates=return_intermediates, context=context, slice_=slice_,
                                  y=y)

    # # # # #

    # # # # # raw_loss process # # # # #
    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown raw_loss type '{loss_type}'")

        assert torch.isfinite(loss).all(), "x_start contains NaN or inf"

        return loss

    # def p_losses(self, x_start, t, context=None, label=None, class_ids=None, y=None, noise=None):
    def p_losses(self, x_start, t, y=None, noise=None, context=None, slice_=None,*args, **kwargs):
        # model out
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
       
        model_out,alpha = self.model(x=x_noisy, timesteps=t, y=y, context=context, slice_=slice_,*args, **kwargs)     # def forward(self, x, timesteps=None, y=None, context=None,**kwargs)
        assert torch.isfinite(model_out).all(), "x_start contains NaN or inf"

        # raw_loss type
        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        # train or val
        log_prefix = 'train' if self.training else 'val'

        # raw_loss.mean
        if len(model_out.shape)==5:
            loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3, 4])
        elif len(model_out.shape)==4:
            loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

#         print(f'model_out:{model_out.shape}  target:{target.shape}')
#         print(f'loss:{loss.shape}')

        assert loss.shape[0] == noise.shape[0]

        # loss_simple from raw_loss.mean
        loss_simple = loss.mean() * self.l_simple_weight
#         print(f'loss_simple:{loss_simple.shape},{loss_simple}')
        assert torch.isfinite(loss_simple).all(), "loss_simple contains NaN or inf"
        loss_dict.update({f'{log_prefix}/loss': loss})
        loss_dict.update({f'{log_prefix}/time': t})
        loss_dict.update({f'{log_prefix}/loss_simple': loss_simple})
        loss_dict.update({f'{log_prefix}/l_simple_weight': self.l_simple_weight})

        # # 由于 original_elbo_weight作为loss_vlb的系数 暂时初始化为0 所以loss_vlb暂时失去作用，于是令：
        total_loss = loss_simple
        loss_dict.update({f'{log_prefix}/total_loss': total_loss})
        
        # # loss_vlb
        # assert self.lvlb_weights[t].shape == raw_loss.shape
        # loss_vlb = (self.lvlb_weights[t] * raw_loss).mean()
        # assert torch.isfinite(loss_vlb).all(), "loss_vlb contains NaN or inf"
        # loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})
        #
        # # total raw_loss from loss_vlb and loss_simple
        # raw_loss = loss_simple + self.original_elbo_weight * loss_vlb
        # assert torch.isfinite(raw_loss).all(), "raw_loss contains NaN or inf"
        # loss_dict.update({f'{log_prefix}/raw_loss': raw_loss})

        # total_loss = l_simple_weight * raw_loss.mean()  + original_elbo_weight * (lvlb_weights[t] * raw_loss).mean()
        #      = 1               * raw_loss.mean()  + 0.                   * (lvlb_weights[t] * raw_loss).mean()
        #      = raw_loss.mean()

        # raw_loss label y
        if y is not None:
            loss_dict.update({f'{log_prefix}/y': y})
#         if log_prefix == 'val':
#             loss_dict.update({f'{log_prefix}/noise': noise})

        return total_loss, loss_dict

    def forward(self, x, device, y=None, context=None, slice_=None, *args, **kwargs):
#         (self, x_start, t, y=None, noise=None, context=None):
#         b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
#         assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=device).long()
        return self.p_losses(x, t, y=y, context=context, slice_=slice_,*args, **kwargs)
