"""多孔介质生成模型训练脚本。

  python train.py --mode vae   --data_dir data-demo/ --epochs 100
  python train.py --mode diff  --data_dir data-demo/ --vae_run runs/vae_z4_e6_XXX/
  python train.py --config my_config.toml          # 手写参数文件
"""

import argparse, os, math, json
from datetime import datetime

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler

import tifffile

from vae import AutoencoderKL_
from diffusion import SP_Model, DDPM_
from data_loader import make_loader

# ===========================================================================
# 工具
# ===========================================================================

def _now():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _save_tif(tensor, path):
    """(1,H,W,D) 或 (H,W,D) torch tensor → 3D tif。"""
    arr = (tensor.squeeze().detach().cpu().numpy() * 255).astype(np.uint8)
    tifffile.imwrite(path, arr)


def _load_vae(device, run_dir=None, ckpt_path=None):
    m = AutoencoderKL_().to(device)
    if run_dir:
        p = json.load(open(f"{run_dir}/params.json"))
        m = AutoencoderKL_(**p["model"]["vae"]).to(device)
    if ckpt_path:
        m.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["model"])
    elif run_dir:
        last = f"{run_dir}/ckpt/vae_last.pt"
        if os.path.exists(last):
            m.load_state_dict(torch.load(last, map_location=device, weights_only=False)["model"])
    return m


def _load_diffusion(device, run_dir=None, ckpt_path=None):
    if run_dir:
        p = json.load(open(f"{run_dir}/params.json"))
        sp = SP_Model(**p["model"]["diffusion"]).to(device)
        ddpm = DDPM_(sfnet=sp, **p["model"]["ddpm"]).to(device)
    else:
        sp = SP_Model().to(device)
        ddpm = DDPM_(sfnet=sp).to(device)
    if ckpt_path:
        ddpm.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["model"])
    elif run_dir:
        last = f"{run_dir}/ckpt/diff_last.pt"
        if os.path.exists(last):
            ddpm.load_state_dict(torch.load(last, map_location=device, weights_only=False)["model"])
    return ddpm


# ===========================================================================
# 学习率
# ===========================================================================

def _build_scheduler(optimizer, warmup_steps, total_steps, lr_min_ratio=0.01):
    def lr_multiplier(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        p = min((step - warmup_steps) / max(1, total_steps - warmup_steps), 1.0)
        return lr_min_ratio + 0.5 * (1 - lr_min_ratio) * (1 + math.cos(math.pi * p))
    return LambdaLR(optimizer, lr_multiplier)


# ===========================================================================
# 保存检查点
# ===========================================================================

def _save_ckpt(model, optimizer, scheduler, scaler, epoch, loss, path):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "epoch": epoch, "loss": loss}, path)


# ===========================================================================
# VAE 训练
# ===========================================================================

def train_vae(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # --- 运行目录 ---
    run_name = args.run_name or f"vae_z{args.vae_z_channels}_e{args.vae_embed_dim}_{_now()}"
    run_dir = f"runs/{run_name}"
    for sub in ["ckpt", "samples"]:
        os.makedirs(f"{run_dir}/{sub}", exist_ok=True)

    # --- 保存配置 ---
    with open(f"{run_dir}/params.json", "w") as f:
        json.dump({
            "run": {"mode": "vae", "name": run_name, "timestamp": _now()},
            "model": {"vae": {"ch": args.vae_ch, "ch_mult": args.vae_ch_mult,
                       "z_channels": args.vae_z_channels, "embed_dim": args.vae_embed_dim,
                       "in_channels": args.vae_in_channels, "out_ch": args.vae_out_ch,
                       "kl_weight": args.vae_kl_weight, "learning_rate": args.lr}},
            "training": {"epochs": args.epochs, "batch_size": args.batch_size,
                          "lr": args.lr, "lr_min_ratio": args.lr_min_ratio,
                          "warmup_epochs": args.warmup_epochs, "save_every": args.save_every},
            "data": {"data_dir": os.path.abspath(args.data_dir)},
        }, f, indent=2)

    # --- 数据 ---
    loader = make_loader(args.data_dir, batch_size=args.batch_size,
                         load_z_porosities=False, load_slice_2d=False)

    # --- 模型 ---
    model = AutoencoderKL_(ch=args.vae_ch, ch_mult=args.vae_ch_mult,
                           embed_dim=args.vae_embed_dim, z_channels=args.vae_z_channels,
                           in_channels=args.vae_in_channels, out_ch=args.vae_out_ch,
                           kl_weight=args.vae_kl_weight, learning_rate=args.lr).to(device)

    opt = model.configure_optimizers()
    total_steps = args.epochs * len(loader)
    scheduler = _build_scheduler(opt, args.warmup_epochs * len(loader), total_steps,
                                 args.lr_min_ratio)
    scaler = GradScaler()
    start_epoch = 1

    # --- 循环 ---
    model.train()
    for epoch in range(start_epoch, args.epochs + 1):
        total_loss, n = 0.0, 0
        for batch in loader:
            x = batch["sample"].to(device)
            with autocast():
                dec, posterior = model(x)
                loss, _ = model.get_loss(x, dec, posterior)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            opt.zero_grad()
            total_loss += loss.item()
            n += 1
        avg = total_loss / n
        print(f"[VAE] epoch {epoch:3d}/{args.epochs}  loss={avg:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            _save_ckpt(model, opt, scheduler, scaler, epoch, avg,
                       f"{run_dir}/ckpt/vae_ep{epoch}.pt")
            # 保存重建样本
            with torch.no_grad():
                recon = model(loader.dataset[0]["sample"].unsqueeze(0).to(device))[0]
            _save_tif(recon, f"{run_dir}/samples/recon_ep{epoch}.tif")
            _save_tif(loader.dataset[0]["sample"],
                      f"{run_dir}/samples/original.tif")

    _save_ckpt(model, opt, scheduler, scaler, args.epochs, avg, f"{run_dir}/ckpt/vae_last.pt")
    print(f"done → {run_dir}")


# ===========================================================================
# Diffusion 训练
# ===========================================================================

def train_diffusion(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # --- 运行目录 ---
    run_name = args.run_name or f"diff_z{args.diff_in_chnl}_t{args.diff_timesteps}_{_now()}"
    run_dir = f"runs/{run_name}"
    for sub in ["ckpt", "samples"]:
        os.makedirs(f"{run_dir}/{sub}", exist_ok=True)

    # --- 加载 VAE ---
    vae = _load_vae(device, args.vae_run, args.vae_ckpt).eval()

    # --- 保存配置 ---
    with open(f"{run_dir}/params.json", "w") as f:
        json.dump({
            "run": {"mode": "diffusion", "name": run_name, "timestamp": _now(),
                    "vae_run": args.vae_run or ""},
            "model": {
                "vae": {"ch": args.vae_ch, "ch_mult": args.vae_ch_mult,
                        "z_channels": args.vae_z_channels, "embed_dim": args.vae_embed_dim},
                "diffusion": {"in_chnl": args.diff_in_chnl, "out_chnl": args.diff_out_chnl,
                              "base_chnl": args.diff_base_chnl, "num_heads": args.diff_num_heads,
                              "chnl_mult": args.diff_chnl_mult, "time_dim": args.diff_time_dim,
                              "att_mult": args.diff_att_mult, "context_dim": args.diff_context_dim,
                              "cond_dim": args.diff_cond_dim},
                "ddpm": {"timesteps": args.diff_timesteps,
                         "linear_start": args.diff_linear_start,
                         "linear_end": args.diff_linear_end,
                         "channels": args.diff_in_chnl,
                         "image_size": 96 // (2 ** (len(args.vae_ch_mult) - 1))},
            },
            "training": {"epochs": args.epochs, "batch_size": args.batch_size,
                          "lr": args.lr, "lr_min_ratio": args.lr_min_ratio,
                          "warmup_epochs": args.warmup_epochs, "save_every": args.save_every},
            "data": {"data_dir": os.path.abspath(args.data_dir)},
        }, f, indent=2)

    # --- 数据 ---
    loader = make_loader(args.data_dir, batch_size=args.batch_size)

    # --- 扩散模型 ---
    sp = SP_Model(in_chnl=args.diff_in_chnl, out_chnl=args.diff_out_chnl,
                  base_chnl=args.diff_base_chnl, num_heads=args.diff_num_heads,
                  chnl_mult=args.diff_chnl_mult, time_dim=args.diff_time_dim,
                  att_mult=args.diff_att_mult, context_dim=args.diff_context_dim,
                  cond_dim=args.diff_cond_dim).to(device)
    ddpm = DDPM_(sfnet=sp, timesteps=args.diff_timesteps,
                 linear_start=args.diff_linear_start,
                 linear_end=args.diff_linear_end).to(device)

    opt = AdamW(ddpm.parameters(), lr=args.lr, betas=(0.9, 0.999))
    total_steps = args.epochs * len(loader)
    scheduler = _build_scheduler(opt, args.warmup_epochs * len(loader), total_steps,
                                 args.lr_min_ratio)
    scaler = GradScaler()

    # --- 循环 ---
    ddpm.train()
    for epoch in range(1, args.epochs + 1):
        total_loss, n = 0.0, 0
        for batch in loader:
            x = batch["sample"].to(device)
            with torch.no_grad():
                z = vae.encode(x).sample()

            por = batch.get("porosity")
            slc = batch.get("slice_2d")
            if por is None or por.numel() == 0:
                por = torch.zeros(x.shape[0], device=device)
            if slc is None:
                slc = x[:, :, 0]
            por, slc = por.to(device), slc.to(device)

            with autocast():
                loss, _ = ddpm(z, device=device, context=por, slice_=slc)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            opt.zero_grad()
            total_loss += loss.item()
            n += 1
        avg = total_loss / n
        print(f"[Diff] epoch {epoch:3d}/{args.epochs}  loss={avg:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            _save_ckpt(ddpm, opt, scheduler, scaler, epoch, avg,
                       f"{run_dir}/ckpt/diff_ep{epoch}.pt")

    _save_ckpt(ddpm, opt, scheduler, scaler, args.epochs, avg, f"{run_dir}/ckpt/diff_last.pt")
    print(f"done → {run_dir}")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 通用
    p.add_argument("--mode", choices=["vae", "diffusion"], default="vae")
    p.add_argument("--config", help="TOML 参数文件 (覆盖默认值)")
    p.add_argument("--run_name", help="自定义运行目录名 (默认自动生成)")
    p.add_argument("--data_dir", default="data-demo/")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_min_ratio", type=float, default=0.01)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--save_every", type=int, default=20)

    # VAE
    p.add_argument("--vae_ch", type=int, default=64)
    p.add_argument("--vae_ch_mult", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--vae_z_channels", type=int, default=4)
    p.add_argument("--vae_embed_dim", type=int, default=6)
    p.add_argument("--vae_in_channels", type=int, default=1)
    p.add_argument("--vae_out_ch", type=int, default=1)
    p.add_argument("--vae_kl_weight", type=float, default=1e-6)

    # Diffusion
    p.add_argument("--vae_run", help="runs/XXX/ (从 params.toml 加载 VAE)")
    p.add_argument("--vae_ckpt", help="VAE .pt 文件")
    p.add_argument("--diff_in_chnl", type=int, default=6)
    p.add_argument("--diff_out_chnl", type=int, default=6)
    p.add_argument("--diff_base_chnl", type=int, default=64)
    p.add_argument("--diff_num_heads", type=int, default=8)
    p.add_argument("--diff_chnl_mult", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--diff_time_dim", type=int, default=256)
    p.add_argument("--diff_att_mult", type=int, nargs="+", default=[4, 2, 1])
    p.add_argument("--diff_context_dim", type=int, default=256)
    p.add_argument("--diff_cond_dim", type=int, default=256)
    p.add_argument("--diff_timesteps", type=int, default=1000)
    p.add_argument("--diff_linear_start", type=float, default=0.0015)
    p.add_argument("--diff_linear_end", type=float, default=0.0195)

    args = p.parse_args()

    # JSON 配置文件覆盖
    if args.config:
        cfg = json.load(open(args.config))
        for section, table in cfg.items():
            for k, v in table.items():
                if hasattr(args, k):
                    setattr(args, k, v)

    if args.mode == "vae":
        train_vae(args)
    else:
        train_diffusion(args)


if __name__ == "__main__":
    main()
