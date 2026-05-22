"""多孔介质生成推理。

  python inference.py --mode vae     --run runs/vae_z4_e6_XXX/
  python inference.py --mode pipeline --run runs/diff_z6_t1000_XXX/
  python inference.py --mode pipeline --vae_ckpt ckpt/vae.pt --diff_ckpt ckpt/diff.pt --data_dir data-demo/
"""

import argparse, os, json
import numpy as np
import torch
from tqdm import tqdm

import tifffile

from vae import AutoencoderKL_
from diffusion import SP_Model, DDPM_
from data_loader import make_loader


# ===========================================================================
# 工具
# ===========================================================================

def _save_tif(tensor, path):
    arr = (tensor.squeeze().detach().cpu().numpy() * 255).astype(np.uint8)
    tifffile.imwrite(path, arr)


def _load_params_json(run_dir):
    return json.load(open(f"{run_dir}/params.json"))


def _build_vae(device, run_dir=None, ckpt_path=None):
    if run_dir:
        cfg = _load_params_json(run_dir)
        # 如果是扩散 run，读取关联的 VAE run
        vae_run = cfg["run"].get("vae_run", "")
        vae_cfg = _load_params_json(vae_run) if vae_run else cfg
        kw = vae_cfg["model"]["vae"]
        m = AutoencoderKL_(**kw).to(device).eval()
        ckpt_dir = vae_run or run_dir
        last = f"{ckpt_dir}/ckpt/vae_last.pt"
        if os.path.exists(last):
            m.load_state_dict(torch.load(last, map_location=device, weights_only=False)["model"])
    else:
        m = AutoencoderKL_().to(device).eval()
    if ckpt_path:
        m.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["model"])
    return m


def _build_diffusion(device, run_dir=None, ckpt_path=None):
    if run_dir:
        cfg = _load_params_json(run_dir)
        dkw = cfg["model"]["diffusion"]
        sp = SP_Model(**dkw).to(device).eval()
        ddpm = DDPM_(sfnet=sp, **cfg["model"]["ddpm"]).to(device).eval()
        last = f"{run_dir}/ckpt/diff_last.pt"
        if os.path.exists(last):
            ddpm.load_state_dict(torch.load(last, map_location=device, weights_only=False)["model"])
    else:
        sp = SP_Model().to(device).eval()
        ddpm = DDPM_(sfnet=sp).to(device).eval()
    if ckpt_path:
        ddpm.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["model"])
    return ddpm


def _save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ===========================================================================
# VAE 重建
# ===========================================================================

def run_vae(args, device):
    vae = _build_vae(device, args.run, args.vae_ckpt)
    data_dir = args.data_dir or _load_params_json(args.run)["data"]["data_dir"]
    loader = make_loader(data_dir, batch_size=1, shuffle=False,
                         load_slice_2d=False, load_z_porosities=False)

    out_dir = f"{args.run}/inference" if args.run else "inference_vae"
    os.makedirs(out_dir, exist_ok=True)

    manifest = []
    for i, batch in enumerate(tqdm(loader, desc="VAE 重建")):
        x = batch["sample"].to(device)
        with torch.no_grad():
            recon, _ = vae(x)

        sub = f"{out_dir}/batch_{i:04d}"
        os.makedirs(sub, exist_ok=True)
        _save_tif(recon, f"{sub}/reconstructed.tif")
        _save_tif(x, f"{sub}/original.tif")
        _save_json(f"{sub}/info.json", {"batch": i, "filename": batch.get("filename", ["unknown"])[0]})
        manifest.append({"batch": i, "dir": f"batch_{i:04d}",
                         "original": batch.get("filename", ["unknown"])[0]})

    _save_json(f"{out_dir}/manifest.json", manifest)
    print(f"done → {out_dir}")


# ===========================================================================
# 扩散生成
# ===========================================================================

def run_pipeline(args, device):
    vae = _build_vae(device, args.run_vae or args.run, args.vae_ckpt)
    ddpm = _build_diffusion(device, args.run, args.diff_ckpt)

    data_dir = args.data_dir
    if not data_dir and args.run:
        data_dir = _load_params_json(args.run)["data"]["data_dir"]

    loader = make_loader(data_dir, batch_size=1, shuffle=False)

    out_dir = f"{args.run}/inference" if args.run else "inference_pipeline"
    os.makedirs(out_dir, exist_ok=True)

    _save_json(f"{out_dir}/inference_params.json", {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "run_dir": args.run or "", "data_dir": data_dir or "",
    })

    manifest = []
    for i, batch in enumerate(tqdm(loader, desc="Pipeline 生成")):
        x = batch["sample"].to(device)
        por = batch.get("porosity")
        slc = batch.get("slice_2d")
        zps = batch.get("z_porosities")
        fname = batch.get("filename", [f"sample_{i:04d}"])[0]

        # z_porosities = 96-element slice-porosity sequence (the condition for diffusion)
        if zps is None or zps.numel() == 0:
            zps = torch.zeros(1, 96, device=device)
        else:
            zps = zps.to(device)
        if slc is None:
            slc = x[:, :, 0]
        slc = slc.to(device)

        sub = f"{out_dir}/batch_{i:04d}"
        os.makedirs(sub, exist_ok=True)

        # 保存条件
        _save_tif(slc, f"{sub}/slice_z0.tif")
        conditions = {"z_porosities": [round(float(v), 6) for v in zps.squeeze().tolist()]}
        if por is not None and por.numel() == 1:
            conditions["porosity_3d"] = round(por.item(), 6)
        _save_json(f"{sub}/conditions.json", conditions)

        # 原生样本
        _save_tif(x, f"{sub}/original.tif")

        # 扩散采样 → 解码
        with torch.no_grad():
            z = ddpm.sample(batch_size=1, context=zps, slice_=slc)
            torch.save(z.cpu(), f"{sub}/latent_z.pt")
            recon = vae.decode(z.to(device))

        _save_tif(recon, f"{sub}/generated.tif")

        manifest.append({"batch": i, "dir": f"batch_{i:04d}",
                         "original": fname,
                         "porosity_3d": conditions.get("porosity_3d")})

    _save_json(f"{out_dir}/manifest.json", manifest)
    print(f"done → {out_dir}")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=["vae", "pipeline"], default="pipeline")
    p.add_argument("--run", help="训练 run 目录 (如 runs/vae_z4_e6_XXXX/)")
    p.add_argument("--run_vae", help="VAE run 目录 (pipeline 模式下单独指定)")
    p.add_argument("--vae_ckpt", help="VAE .pt 文件")
    p.add_argument("--diff_ckpt", help="Diffusion .pt 文件")
    p.add_argument("--data_dir", help="数据目录 (覆盖 run 中的数据路径)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    if args.mode == "vae":
        run_vae(args, device)
    else:
        run_pipeline(args, device)


if __name__ == "__main__":
    main()
