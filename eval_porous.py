"""多孔介质生成质量评估。

原子操作:
  process_folder(path, ...)         处理单个文件夹 → 原始数据
  summarize(raw, label, save_to)    原始数据 → 统计 YAML
  compare(one, two, save_dir, ...)  两份统计 → 对比图

便捷组合:
  run_all(train_path, gen_path, ..., save_dir)  一键: 处理+统计+画图

指标:
  phi     孔隙度 / S₂(r) 两点相关函数 / L(r) 线性路径函数
"""

import os, glob, yaml
import numpy as np
import tifffile
import scipy.io as sio
from scipy import fft
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple


# ===========================================================================
# 核心算法
# ===========================================================================

def __radial_average(autocorr, shape):
    z, y, x = np.indices(shape)
    center = np.array(shape) // 2
    r = np.sqrt((x - center[2]) ** 2 + (y - center[1]) ** 2 + (z - center[0]) ** 2)
    r = r.astype(int)

    r_flat = r.ravel()
    autocorr_flat = autocorr.ravel()

    r_max = min(shape) // 2
    r_values = np.arange(0, r_max)
    tpf = np.zeros(r_max)
    for radius in range(r_max):
        mask = (r_flat == radius)
        if np.any(mask):
            tpf[radius] = np.mean(autocorr_flat[mask])
    return tpf, r_values


def __calculate_tpf(img):
    """两点相关函数 S₂(r)。Zero-padding → FFT → autocorrelation → radial average。"""
    padded_shape = tuple(2 * dim for dim in img.shape)
    padded_img = np.zeros(padded_shape, dtype=img.dtype)
    padded_img[tuple(slice(0, dim) for dim in img.shape)] = img

    F = fft.fftn(padded_img)
    autocorr = fft.ifftn(F * np.conj(F)).real / img.size

    autocorr = fft.fftshift(autocorr)
    center = tuple(slice(dim // 2, dim + dim // 2) for dim in img.shape)
    autocorr = autocorr[center]

    return __radial_average(autocorr, img.shape)


def __calculate_lpf(img):
    """线性路径函数 L(r)。6 方向 chord-length 统计。"""
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    all_lengths = []

    for dx, dy, dz in directions:
        lengths = np.zeros_like(img, dtype=np.uint8)
        z_range = range(img.shape[0] - 1, -1, -1) if dz < 0 else range(img.shape[0])
        y_range = range(img.shape[1] - 1, -1, -1) if dy < 0 else range(img.shape[1])
        x_range = range(img.shape[2] - 1, -1, -1) if dx < 0 else range(img.shape[2])

        for z in z_range:
            for y in y_range:
                for x in x_range:
                    if img[z, y, x] == 0:
                        lengths[z, y, x] = 0
                        continue
                    prev_z, prev_y, prev_x = z - dz, y - dy, x - dx
                    if 0 <= prev_z < img.shape[0] and 0 <= prev_y < img.shape[1] and 0 <= prev_x < img.shape[2]:
                        lengths[z, y, x] = lengths[prev_z, prev_y, prev_x] + 1
                    else:
                        lengths[z, y, x] = 1
        all_lengths.extend(lengths[img == 1].ravel())

    if not all_lengths:
        return np.array([]), np.array([])
    max_len = max(all_lengths)
    l_values = np.arange(1, max_len + 1)
    total = len(all_lengths)
    lpf = np.array([np.sum(np.array(all_lengths) >= L) / total for L in l_values])
    return lpf, l_values


def __load_and_binarize(image_path):
    """读取 .tif/.mat 图像，二值化为 {0,1}。"""
    ext = os.path.splitext(image_path)[1].lower()
    if ext in ('.tif', '.tiff'):
        img = tifffile.imread(image_path)
    elif ext == '.mat':
        img = sio.loadmat(image_path)['BW']
    else:
        raise ValueError(f"不支持的文件格式: {ext}")
    img = img.astype(np.float32)
    img = np.where(img > 127, 1, 0) if np.max(img) > 1 else np.where(img > 0.5, 1, 0)
    if img.shape != (96, 96, 96):
        raise ValueError(f"图像尺寸错误: 预期 (96,96,96), 实际 {img.shape}")
    return img


def __process_single(image_path):
    """单张图像: 返回 (tpf, r_values, lpf, l_values, phi)。"""
    img = __load_and_binarize(image_path)
    phi = np.mean(img)
    tpf, r_values = __calculate_tpf(img)
    lpf, l_values = __calculate_lpf(img)
    return tpf, r_values, lpf, l_values, phi


# ===========================================================================
# 批量处理
# ===========================================================================

def __list_files(folder_path):
    files = []
    files.extend(glob.glob(os.path.join(folder_path, '*.[Tt][Ii][Ff]')))
    files.extend(glob.glob(os.path.join(folder_path, '*.[Tt][Ii][Ff][Ff]')))
    files.extend(glob.glob(os.path.join(folder_path, '*.[Mm][Aa][Tt]')))
    return list(set(files))


def __match(filename, include_key, exclude_keys, case_sensitive):
    fn = filename if case_sensitive else filename.lower()
    if include_key is not None:
        inc = include_key if case_sensitive else include_key.lower()
        if inc not in fn:
            return False
    if exclude_keys:
        exc = [k if case_sensitive else k.lower() for k in exclude_keys]
        if any(e in fn for e in exc):
            return False
    return True


def process_folder(folder_path, include_key=None, exclude_keys=None,
                   case_sensitive=False, sample_limit=None):
    """处理单个文件夹，返回原始数据。

    返回:
        dict {"tpf": list[1D array], "r": 1D array, "lpf": list[1D array],
              "l": 1D array, "phi": list[float]}
    """
    all_tpf, all_lpf, all_phi = [], [], []
    all_r, all_l = None, None
    collected = 0

    for file in tqdm(__list_files(folder_path), desc=os.path.basename(folder_path)):
        if not __match(os.path.basename(file), include_key, exclude_keys, case_sensitive):
            continue
        try:
            tpf, r, lpf, l, phi = __process_single(file)
            if all_r is None: all_r = r
            if all_l is None or len(l) > len(all_l): all_l = l
            all_tpf.append(tpf); all_lpf.append(lpf); all_phi.append(phi)
            collected += 1
            if sample_limit and collected >= sample_limit:
                print(f"已收集 {collected} 个样本（达到限制），停止处理")
                break
        except (ValueError, Exception) as e:
            print(f"跳过 {os.path.basename(file)}: {e}")

    if all_l is not None and all_lpf:
        max_l = len(all_l)
        for i in range(len(all_lpf)):
            if len(all_lpf[i]) < max_l:
                all_lpf[i] = np.pad(all_lpf[i], (0, max_l - len(all_lpf[i])), 'constant')
    return {"tpf": all_tpf, "r": all_r, "lpf": all_lpf, "l": all_l, "phi": all_phi}


def process_all(parent_path, folder_pattern, include_key=None, exclude_keys=None,
                case_sensitive=False, sample_limit=None):
    """遍历 parent_path 下匹配 pattern 的子文件夹，分别处理后合并。

    使用场景：生成集通常在父目录下有多个子文件夹如 fake_set_seed0/, fake_set_seed1/。
              调用 process_all("ckpt/fake_set_eval", "fake_set_seed*", ...) 即可。

    返回:
        合并后的原始数据 dict（同 process_folder 返回值）。
    """
    folders = glob.glob(os.path.join(parent_path, folder_pattern))
    if not folders:
        raise ValueError(f"未找到匹配 '{folder_pattern}' 的子文件夹于 {parent_path}")

    merged = {"tpf": [], "r": None, "lpf": [], "l": None, "phi": []}
    collected = 0
    for folder in tqdm(folders, desc=f"遍历 {folder_pattern}"):
        if sample_limit and collected >= sample_limit:
            break
        limit = sample_limit - collected if sample_limit else None
        raw = process_folder(folder, include_key, exclude_keys, case_sensitive, limit)
        merged["tpf"].extend(raw["tpf"])
        merged["lpf"].extend(raw["lpf"])
        merged["phi"].extend(raw["phi"])
        collected += len(raw["tpf"])
        if merged["r"] is None: merged["r"] = raw["r"]
        if merged["l"] is None or (raw["l"] is not None and len(raw["l"]) > len(merged["l"])):
            merged["l"] = raw["l"]
    return merged


# ===========================================================================
# 统计计算
# ===========================================================================

def __load_yaml(path):
    with open(path, encoding='utf-8') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def __compute_corr_stats(data_list, x_values):
    if not data_list:
        return {"mean": [], "std": [], "min": [], "max": [], "x": []}
    arr = np.array(data_list)
    return {"mean": np.mean(arr, 0).tolist(), "std": np.std(arr, 0).tolist(),
            "min": np.min(arr, 0).tolist(), "max": np.max(arr, 0).tolist(),
            "x": x_values.tolist() if hasattr(x_values, 'tolist') else list(x_values)}


def __compute_phi_stats(phi_list):
    if not phi_list: return {}
    arr = np.array(phi_list)
    return {k: float(v) for k, v in zip(
        ["mean", "std", "min", "max", "25%", "50%", "75%"],
        [np.mean(arr), np.std(arr), np.min(arr), np.max(arr),
         np.percentile(arr, 25), np.percentile(arr, 50), np.percentile(arr, 75)])}


def summarize(raw, label="", save_to=None):
    """单份原始数据 → 统计值字典，可选写入 YAML。

    参数:
        raw:     process_folder() 或 process_all() 返回的原始数据
        label:   标签（会写入 YAML 的 parameters.name）
        save_to: 可选，YAML 保存路径
    返回:
        dict: 可直接传给 compare() 或手动保存
    """
    stats = {
        "parameters": {
            "name": label,
            "phi_stats": __compute_phi_stats(raw["phi"]),
            "save_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "statistics": {
            "tpf":      __compute_corr_stats(raw["tpf"], raw["r"]),
            "norm_tpf": __compute_corr_stats([t / p for t, p in zip(raw["tpf"], raw["phi"])], raw["r"]),
            "lpf":      __compute_corr_stats(raw["lpf"], raw["l"]),
        },
    }
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        with open(save_to, "w", encoding="utf-8") as f:
            yaml.dump(stats, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        print(f"统计已保存: {save_to}")
    return stats


# ===========================================================================
# 对比画图
# ===========================================================================

def __ensure_list(v):
    """兼容: 传入 YAML 路径(str)、stats dict 或 list of paths/dicts。"""
    if isinstance(v, (list, tuple)):
        return [__load_yaml(x) if isinstance(x, str) else x for x in v]
    return [__load_yaml(v) if isinstance(v, str) else v]


def __plot_boxplot(all_stats, all_labels, save_path):
    n = len(all_stats)
    colors = ['#1E88E5'] + [plt.cm.tab10(i)[:3] for i in range(1, n)]
    plt.figure(figsize=(8, 6))
    bp = plt.boxplot(all_stats, positions=list(range(n)), widths=0.5,
                     patch_artist=True, showfliers=False,
                     medianprops={"color": "white", "linewidth": 2},
                     whiskerprops={"color": "black", "linewidth": 1.5},
                     capprops={"color": "black", "linewidth": 1.5})
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_edgecolor('black')
    plt.xticks(range(n), all_labels, fontsize=12)
    plt.ylabel("Porosity ($\\phi$)", fontsize=12)
    plt.title("Porosity Distribution Comparison", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=800, bbox_inches="tight"); plt.show()
    print(f"已保存: {save_path}")


def __plot_curves(series_list, save_path, xlabel, ylabel, title):
    """series_list: [{"x": ..., "mean": ..., "std": ..., "label": ..., "color": ...}, ...]"""
    plt.figure(figsize=(12, 6))
    for s in series_list:
        plt.plot(s["x"], s["mean"], label=s.get("label", ""), color=s.get("color", None), linewidth=2)
        if s.get("std") is not None:
            plt.fill_between(s["x"], np.array(s["mean"]) - np.array(s["std"]),
                             np.array(s["mean"]) + np.array(s["std"]),
                             color=s.get("fill_color", s.get("color", "gray")), alpha=0.3)
    plt.xlabel(xlabel, fontsize=12); plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14); plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.7); plt.tight_layout()
    plt.savefig(save_path, dpi=800, bbox_inches="tight"); plt.show()
    print(f"已保存: {save_path}")


def compare(one, two, save_dir="eval_results",
            label_one="A", label_two="B",
            color_one="#1E88E5", color_two="#E53935"):
    """对比两份统计结果并画图。

    参数:
        one, two:  YAML 路径(str) 或 summarize() 返回的 dict。
                    one 作为参考（通常在图上标为左侧/蓝色），two 作为对比。
        save_dir:  图片输出目录
        label_one/two: 图例标签
    返回:
        None
    """
    os.makedirs(save_dir, exist_ok=True)
    a, b = __ensure_list(one)[0], __ensure_list(two)[0]

    # 孔隙度箱式图
    __plot_boxplot(
        [tuple(a["parameters"]["phi_stats"][k] for k in ["min","25%","50%","75%","max"]),
         tuple(b["parameters"]["phi_stats"][k] for k in ["min","25%","50%","75%","max"])],
        [label_one, label_two],
        os.path.join(save_dir, "porosity_boxplot.png"))

    # TPF / 归一化 TPF / LPF
    for key, xkey, xl, yl in [
        ("tpf", "x", "Distance", "$S_2(r)$"),
        ("norm_tpf", "x", "Distance", "$S_2(r)/\\phi$"),
        ("lpf", "x", "Path Length", "LPF Probability"),
    ]:
        __plot_curves([
            dict(x=a["statistics"][key]["x"], mean=a["statistics"][key]["mean"],
                 std=a["statistics"][key]["std"], label=label_one, color=color_one,
                 fill_color=color_one),
            dict(x=b["statistics"][key]["x"], mean=b["statistics"][key]["mean"],
                 std=b["statistics"][key]["std"], label=label_two, color=color_two,
                 fill_color=color_two),
        ], os.path.join(save_dir, f"{key}_comparison.png"), xl, yl,
           f"{'Normalized ' if 'norm' in key else ''}{'Two-Point Correlation' if 'tpf' in key else 'Linear Path'} Function")


def compare_multi(one, others, save_dir="multi_eval",
                  label_one="Train", other_labels=None,
                  color_one="#1E88E5"):
    """一份参考 vs 多份对比。

    参数:
        one:    参考 YAML/stat dict（如训练集统计）
        others: 对比项列表（每个元素是 YAML 路径或 stat dict）
    """
    os.makedirs(save_dir, exist_ok=True)
    ref = __ensure_list(one)[0]
    others = __ensure_list(others)
    if other_labels is None:
        other_labels = [ref["parameters"].get("name", f"Model_{i + 1}") for i in range(len(others))]

    # 箱式图
    keys = ["min", "25%", "50%", "75%", "max"]
    all_stats = [tuple(ref["parameters"]["phi_stats"][k] for k in keys)]
    all_stats += [tuple(o["parameters"]["phi_stats"][k] for k in keys) for o in others]
    __plot_boxplot(all_stats, [label_one] + other_labels,
                   os.path.join(save_dir, "multi_porosity_boxplot.png"))

    # 相关函数
    for key, xkey, xl, yl in [
        ("tpf", "x", "Distance", "$S_2(r)$"),
        ("norm_tpf", "x", "Distance", "$S_2(r)/\\phi$"),
        ("lpf", "x", "Path Length", "LPF Probability"),
    ]:
        series = [dict(x=ref["statistics"][key]["x"], mean=ref["statistics"][key]["mean"],
                       std=ref["statistics"][key]["std"], label=label_one, color=color_one,
                       fill_color=color_one)]
        for i, o in enumerate(others):
            series.append(dict(x=o["statistics"][key]["x"], mean=o["statistics"][key]["mean"],
                               std=o["statistics"][key]["std"], label=other_labels[i]))
        __plot_curves(series, os.path.join(save_dir, f"multi_{key}_comparison.png"),
                      xl, yl, f"Multi-Model {'Normalized ' if 'norm' in key else ''}{'TPF' if 'tpf' in key else 'LPF'} Comparison")


# ===========================================================================
# 便捷组合
# ===========================================================================

def run_all(train_path, gen_path, save_dir="eval_results",
            label_train="Train", label_gen="Generated",
            gen_is_parent=False, gen_folder_pattern="fake_set_seed*",
            include_key=None, exclude_keys=None,
            case_sensitive=False, sample_limit=None):
    """一键: 处理训练集 + 处理生成集 → 统计 → 对比画图。

    参数:
        train_path:       训练集文件夹路径
        gen_path:         生成集路径（单个文件夹 或 父目录）
        gen_is_parent:    True 表示 gen_path 是父目录，会自动扫描 gen_folder_pattern 子文件夹
        gen_folder_pattern: 当 gen_is_parent=True 时，子文件夹匹配模式
        include_key:      文件名关键字（两集共用；如需不同，分别调 process_folder + summarize + compare）
        exclude_keys:     排除关键字（同上）
    """
    os.makedirs(save_dir, exist_ok=True)

    print(">>> 处理训练集")
    train_raw = process_folder(train_path, include_key, exclude_keys, case_sensitive, sample_limit)
    train_yaml = os.path.join(save_dir, "stats_train.yaml")
    summarize(train_raw, label=label_train, save_to=train_yaml)

    print(">>> 处理生成集")
    if gen_is_parent:
        gen_raw = process_all(gen_path, gen_folder_pattern, include_key, exclude_keys,
                              case_sensitive, sample_limit)
    else:
        gen_raw = process_folder(gen_path, include_key, exclude_keys, case_sensitive, sample_limit)
    gen_yaml = os.path.join(save_dir, "stats_gen.yaml")
    summarize(gen_raw, label=label_gen, save_to=gen_yaml)

    print(">>> 画对比图")
    compare(train_yaml, gen_yaml, save_dir, label_one=label_train, label_two=label_gen)


# ===========================================================================
# 用法示例
# ===========================================================================

if __name__ == "__main__":
    # ---- 最小使用: 只算一个文件夹 ----
    # raw = process_folder("my_samples/", include_key="SPUNet")
    # summarize(raw, label="SPUNet", save_to="stats_SPUNet.yaml")

    # ---- 处理多个子文件夹 ----
    # raw = process_all("ckpt/eval", "fake_set_seed*", include_key="SPUNet")
    # summarize(raw, label="SPUNet_all", save_to="stats_SPUNet.yaml")

    # ---- 对比两个 YAML ----
    # compare("stats_train.yaml", "stats_gen.yaml", save_dir="eval_results",
    #         label_one="Train", label_two="Generated")

    # ---- 一键 ----
    # run_all(
    #     train_path       = "DATA/VAE_DATA/AIMAX_all/mat_cls5_train",
    #     gen_path         = "Ckpt_Slice_ldm/Dim_ckpt_SPUNet_try_4nd_ep",
    #     gen_is_parent    = True,
    #     gen_folder_pattern = "fake_set_seed*",
    #     save_dir         = "eval_results",
    #     include_key      = "SPUNet",
    #     exclude_keys     = ["raw"],
    #     label_train      = "Train",
    #     label_gen        = "SPUNet",
    # )

    # ---- 多模型对比 ----
    # compare_multi("stats_train.yaml",
    #               ["stats_modelA.yaml", "stats_modelB.yaml"],
    #               other_labels=["ModelA", "ModelB"])
    pass
