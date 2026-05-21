"""验证模型可跑通: shape → 梯度 → 采样 → pipeline.
4050 6GB 足够，CPU 会很慢但也能跑。
"""
import gc
import time
import torch

from vae import AutoencoderKL_
from diffusion import SP_Model, DDPM_

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _info(msg):
    t = time.strftime("%H:%M:%S")
    gb = ""
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        total = getattr(p, 'total_memory', getattr(p, 'total_mem', 0)) / 1024**3
        gb = f" | GPU {torch.cuda.memory_allocated() / 1024**3:.1f}/{total:.0f}GB"
    print(f"[{t}] {msg}{gb}")


def test_vae_shape():
    """前向 shape，~0.5GB，几秒"""
    _info("1. VAE shape check")
    model = AutoencoderKL_().to(DEVICE)
    n = sum(p.numel() for p in model.parameters())
    print(f"    params: {n / 1e6:.1f}M")
    x = torch.randn(1, 1, 96, 96, 96, device=DEVICE)
    with torch.no_grad():
        dec, posterior = model(x)
    print(f"    input : {tuple(x.shape)}")
    print(f"    output: {tuple(dec.shape)}")
    assert dec.shape == x.shape
    del model, x, dec, posterior
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    _info("    PASSED")


def test_vae_grad():
    """反向传播，~2-4GB，几十秒"""
    _info("2. VAE backward")
    model = AutoencoderKL_().to(DEVICE)
    model.train()
    x = torch.randn(1, 1, 96, 96, 96, device=DEVICE)
    dec, posterior = model(x)
    loss, _ = model.get_loss(x, dec, posterior)
    loss.backward()
    no_g = [n for n, p in model.named_parameters() if p.grad is None and p.requires_grad]
    if no_g:
        for n in no_g:
            print(f"    WARNING: no grad for {n}")
    print(f"    loss: {loss.item():.4f}, grads OK")
    del model, x, dec, posterior, loss
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    _info("    PASSED")


def test_diffusion_shape():
    """前向 shape，~0.1GB，瞬间"""
    _info("3. Diffusion shape check")
    sp = SP_Model().to(DEVICE)
    ddpm = DDPM_(sfnet=sp).to(DEVICE)
    n = sum(p.numel() for p in ddpm.parameters())
    print(f"    params: {n / 1e6:.1f}M")
    z = torch.randn(2, 6, 12, 12, 12, device=DEVICE)
    p = torch.rand(2, device=DEVICE) * 0.5
    s = torch.randn(2, 1, 96, 96, device=DEVICE)
    with torch.no_grad():
        loss, _ = ddpm(z, device=DEVICE, context=p, slice_=s)
    print(f"    loss (no grad): {loss.item():.4f}")
    del sp, ddpm, z, p, s, loss
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    _info("    PASSED")


def test_diffusion_grad():
    """反向，~0.3GB，瞬间"""
    _info("4. Diffusion backward")
    sp = SP_Model().to(DEVICE)
    ddpm = DDPM_(sfnet=sp).to(DEVICE)
    ddpm.train()
    z = torch.randn(1, 6, 12, 12, 12, device=DEVICE)
    p = torch.rand(1, device=DEVICE) * 0.5
    s = torch.randn(1, 1, 96, 96, device=DEVICE)
    loss, _ = ddpm(z, device=DEVICE, context=p, slice_=s)
    loss.backward()
    print(f"    loss: {loss.item():.4f}, grads OK")
    del sp, ddpm, z, p, s, loss
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    _info("    PASSED")


def test_diffusion_sample():
    """采样（只用 20 步验证流程），秒级"""
    _info("5. Diffusion sampling (20 steps)")
    sp = SP_Model().to(DEVICE)
    ddpm = DDPM_(sfnet=sp, timesteps=20).to(DEVICE).eval()
    p = torch.rand(1, device=DEVICE) * 0.5
    s = torch.randn(1, 1, 96, 96, device=DEVICE)
    with torch.no_grad():
        out = ddpm.sample(batch_size=1, context=p, slice_=s)
    print(f"    sample: {tuple(out.shape)}")
    assert out.shape == (1, 6, 12, 12, 12)
    del sp, ddpm
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    _info("    PASSED")


def test_pipeline():
    """端到端: encode → sample → decode（全 no_grad）"""
    _info("6. Pipeline (20 steps)")
    vae = AutoencoderKL_().to(DEVICE).eval()
    sp = SP_Model().to(DEVICE)
    ddpm = DDPM_(sfnet=sp, timesteps=20).to(DEVICE).eval()
    x = torch.randn(1, 1, 96, 96, 96, device=DEVICE)
    p = torch.rand(1, device=DEVICE) * 0.5
    s = torch.randn(1, 1, 96, 96, device=DEVICE)
    with torch.no_grad():
        z = vae.encode(x).sample()
        z_sample = ddpm.sample(batch_size=1, context=p, slice_=s)
        recon = vae.decode(z_sample)
    print(f"    {tuple(x.shape)} → encode → {tuple(z.shape)}")
    print(f"    sample → {tuple(z_sample.shape)} → decode → {tuple(recon.shape)}")
    assert recon.shape == (1, 1, 96, 96, 96)
    del vae, sp, ddpm
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    _info("    PASSED")


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Memory-heavy tests: 2 (VAE backward ~2-4GB), 4 (Diff backward ~0.3GB)")
    print(f"Safe tests if OOM: 1, 3, 5, 6\n")
    test_vae_shape()
    test_diffusion_shape()
    test_diffusion_sample()
    test_pipeline()
    test_vae_grad()
    test_diffusion_grad()
    _info("All tests passed.")
