"""多孔介质数据加载器 — 支持 .mat 和 .tif 两种格式。

可加载项（按需计算，默认全部加载）:
  sample        3D 二值化样本 (1, 96, 96, 96)
  slice_2d      第一个 z 切片 (1, 96, 96)
  porosity      3D 孔隙度 (标量)
  z_porosities  96 个 2D 切片的孔隙度序列 (96,)

用法:
  ds = PorousDataset("data-demo/")                    # 默认全部加载
  ds = PorousDataset("data-demo/", load_z_porosities=False)  # 不计算z序列
  dl = DataLoader(ds, batch_size=4, shuffle=True)
"""

import os
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import tifffile as tiff
    HAS_TIFF = True
except ImportError:
    HAS_TIFF = False


class PorousDataset(Dataset):
    def __init__(self, data_dir,
                 load_slice_2d=True,
                 load_porosity=True,
                 load_z_porosities=True):
        """
        data_dir:          文件夹路径，自动发现 .mat 和 .tif
        load_slice_2d:     是否计算第一个 z 切片
        load_porosity:     是否计算 3D 孔隙度
        load_z_porosities: 是否计算 96 个 z 切片孔隙度序列
        """
        self.data_dir = data_dir
        self.load_slice_2d = load_slice_2d
        self.load_porosity = load_porosity
        self.load_z_porosities = load_z_porosities

        self.files = []
        for f in sorted(os.listdir(data_dir)):
            ext = os.path.splitext(f)[1].lower()
            if ext in ('.mat', '.tif', '.tiff'):
                if ext in ('.tif', '.tiff') and not HAS_TIFF:
                    continue
                self.files.append(f)

        if not self.files:
            raise FileNotFoundError(f"{data_dir} 中未找到 .mat 或 .tif 文件")

    def __len__(self):
        return len(self.files)

    def _load_mat(self, path):
        """加载 .mat (字段 'BW') → (96, 96, 96) float32 二值化。"""
        data = sio.loadmat(path)
        key = 'BW' if 'BW' in data else [k for k in data if not k.startswith('__')][0]
        img = data[key].astype(np.float32)
        return self._binarize(img)

    def _load_tif(self, path):
        """加载 .tif → (96, 96, 96) float32 二值化。"""
        img = tiff.imread(path).astype(np.float32)
        return self._binarize(img)

    @staticmethod
    def _binarize(img):
        """像素值 → {0, 1}。"""
        return (img > (127 if img.max() > 1 else 0.5)).astype(np.float32)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.files[idx])
        ext = os.path.splitext(path)[1].lower()

        sample = self._load_mat(path) if ext == '.mat' else self._load_tif(path)
        # sample: (96, 96, 96) float32, values in {0, 1}

        result = {"sample": torch.from_numpy(sample).unsqueeze(0),  # (1, 96, 96, 96)
                  "filename": self.files[idx]}

        if self.load_slice_2d:
            result["slice_2d"] = torch.from_numpy(sample[0]).unsqueeze(0)  # (1, 96, 96)

        if self.load_porosity:
            result["porosity"] = torch.tensor(sample.mean(), dtype=torch.float32)

        if self.load_z_porosities:
            z_phis = sample.mean(axis=(1, 2))  # (96,)
            result["z_porosities"] = torch.from_numpy(z_phis).float()

        return result


def make_loader(data_dir, batch_size=4, shuffle=True, num_workers=0, **kwargs):
    """快捷创建 DataLoader。kwargs 传给 PorousDataset。"""
    ds = PorousDataset(data_dir, **kwargs)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


if __name__ == "__main__":
    # 测试: 默认加载全部
    ds = PorousDataset("data-demo")
    print(f"文件数: {len(ds)}")
    item = ds[0]
    for k, v in item.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: shape={tuple(v.shape)}")
        else:
            print(f"  {k}: {v.item():.4f}")
