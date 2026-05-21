# 3D Porous Media Generation — Usage Guide

## Installation

```bash
conda create -n porous python=3.10 -y && conda activate porous
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

## Project Structure

```
nn_utils.py        nn_blocks.py        vae.py        diffusion.py      ← Model definitions
train.py           data_loader.py      inference.py                     ← Training / Inference
eval_porous.py     test_models.py      requirements.txt                 ← Evaluation / Testing
```

## Data Preparation

Place all training samples in a single directory. Two formats are supported and can be mixed:

```
my_data/
├── rock_001.mat      ← .mat file, with variable 'BW' of shape (96, 96, 96)
├── rock_002.mat
├── rock_003.tif      ← .tif file, 3D image of shape (96, 96, 96)
└── ...
```

- `.mat` : reads the `BW` field automatically
- `.tif` : loaded via `tifffile.imread()`
- Binary threshold applied automatically (127 for 0-255 images, 0.5 for 0-1 images)
- All samples must be 96×96×96

---

## 1. Training

### VAE Stage

```bash
python train.py --mode vae --data_dir my_data/ --epochs 200
```

Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_dir` | data-demo/ | Path to training data |
| `--epochs` | 100 | Number of training epochs |
| `--batch_size` | 1 | Batch size |
| `--lr` | 1e-4 | Peak learning rate |
| `--lr_min_ratio` | 0.01 | Minimum lr / peak lr ratio |
| `--warmup_epochs` | 5 | Warmup duration in epochs |
| `--save_every` | 20 | Save checkpoint every N epochs |
| `--vae_z_channels` | 4 | Latent variable channels |
| `--vae_embed_dim` | 6 | Latent embedding dimension |
| `--vae_kl_weight` | 1e-6 | KL divergence weight |
| `--run_name` | auto | Custom run directory name |

Output directory:

```
runs/vae_z4_e6_20260522_143000/
├── params.json           ← Full parameter record
├── ckpt/
│   ├── vae_ep20.pt       ← Periodic checkpoint
│   └── vae_last.pt       ← Final checkpoint
└── samples/
    ├── original.tif       ← First sample from training set
    └── recon_ep*.tif      ← Reconstructions at saved epochs
```

### Diffusion Stage

```bash
python train.py --mode diffusion --data_dir my_data/ --epochs 500 \
    --vae_run runs/vae_z4_e6_20260522_143000/
```

Key parameters (shared with VAE, plus):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--vae_run` | — | VAE run directory (**required**) |
| `--diff_timesteps` | 1000 | Diffusion steps |
| `--diff_cond_dim` | 256 | Condition embedding dimension |

Output directory:

```
runs/diff_z6_t1000_20260522_150000/
├── params.json
├── ckpt/
│   ├── diff_ep100.pt
│   └── diff_last.pt
└── samples/
```

---

## 2. Inference

### VAE Reconstruction

```bash
python inference.py --mode vae --run runs/vae_z4_e6_20260522_143000/
```

```
inference/
├── batch_000/
│   ├── original.tif
│   ├── reconstructed.tif
│   └── info.json
├── batch_001/ ...
└── manifest.json
```

### Latent Diffusion Generation (Pipeline)

```bash
python inference.py --mode pipeline --run runs/diff_z6_t1000_20260522_150000/
```

```
inference/
├── inference_params.json
├── batch_000/
│   ├── slice_z0.tif        ← Condition slice
│   ├── conditions.json     ← Porosity conditions
│   ├── original.tif        ← Ground-truth sample
│   ├── latent_z.pt         ← Latent variable (6, 12, 12, 12)
│   └── generated.tif       ← Generated result
├── batch_001/ ...
└── manifest.json
```

---

## 3. Evaluation

```python
from eval_porous import process_folder, process_all, summarize, compare

# Process a single folder
raw = process_folder("my_data/", include_key="rock")
summarize(raw, label="real_data", save_to="stats_real.yaml")

# Process generated batches
raw = process_all("runs/diff_z6_XXX/inference", "batch_*")
summarize(raw, label="generated", save_to="stats_gen.yaml")

# Compare
compare("stats_real.yaml", "stats_gen.yaml", save_dir="eval_results",
        label_one="Train", label_two="Generated")
```

Produces 4 comparison figures: porosity boxplot, S₂(r), normalized S₂(r), and L(r).

---

## 4. Launching from a Config File

```bash
# Write my_config.json:
{
  "run": {"mode": "vae"},
  "model": {"vae": {"ch": 64, "z_channels": 4, "embed_dim": 6}},
  "training": {"epochs": 200, "lr": 1e-4},
  "data": {"data_dir": "my_data/"}
}

python train.py --config my_config.json
```

CLI arguments override their JSON counterparts when both are provided.
