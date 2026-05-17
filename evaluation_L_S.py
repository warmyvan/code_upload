import os
import glob
import numpy as np
import tifffile
import scipy.io as sio
from scipy import fft
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml
from datetime import datetime


# ------------------- 基础计算函数（不变） ------------------- #
def radial_average(autocorr, shape):
    z, y, x = np.indices(shape)
    center = np.array(shape) // 2
    r = np.sqrt((x - center[2])**2 + (y - center[1])**2 + (z - center[0])**2)
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

def calculate_tpf(img, phi):
    padded_shape = tuple(2 * dim for dim in img.shape)
    padded_img = np.zeros(padded_shape, dtype=img.dtype)
    slices = tuple(slice(0, dim) for dim in img.shape)
    padded_img[slices] = img
    
    F = fft.fftn(padded_img)
    autocorr = fft.ifftn(F * np.conj(F)).real
    autocorr = autocorr / (img.size * phi**2)
    
    center_slices = tuple(slice(0, dim) for dim in img.shape)
    autocorr = autocorr[center_slices]
    
    tpf, r_values = radial_average(autocorr, img.shape)
    return tpf, r_values

def calculate_lpf(img):
    directions = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
    all_lengths = []
    
    for dx, dy, dz in directions:
        lengths = np.zeros_like(img, dtype=np.uint8)
        z_range = range(img.shape[0]-1, -1, -1) if dz < 0 else range(img.shape[0])
        y_range = range(img.shape[1]-1, -1, -1) if dy < 0 else range(img.shape[1])
        x_range = range(img.shape[2]-1, -1, -1) if dx < 0 else range(img.shape[2])
        
        for z in z_range:
            for y in y_range:
                for x in x_range:
                    if img[z, y, x] == 0:
                        lengths[z, y, x] = 0
                        continue
                    prev_z, prev_y, prev_x = z - dz, y - dy, x - dx
                    if (0 <= prev_z < img.shape[0] and 
                        0 <= prev_y < img.shape[1] and 
                        0 <= prev_x < img.shape[2]):
                        lengths[z, y, x] = lengths[prev_z, prev_y, prev_x] + 1
                    else:
                        lengths[z, y, x] = 1
        all_lengths.extend(lengths[img == 1].ravel())
    
    if not all_lengths:
        return np.array([]), np.array([])
    max_length = max(all_lengths)
    l_values = np.arange(1, max_length + 1)
    total_paths = len(all_lengths)
    lpf = np.zeros(max_length + 1)
    for L in l_values:
        lpf[L] = np.sum(np.array(all_lengths) >= L) / total_paths
    return lpf[1:], l_values

def calculate_tpf_lpf(image_path):
    ext = os.path.splitext(image_path)[1].lower()
    
    if ext in ('.tif', '.tiff'):
        img = tifffile.imread(image_path)
    elif ext == '.mat':
        try:
            mat_data = sio.loadmat(image_path)
            img = mat_data['BW']
        except KeyError:
            raise ValueError(f"MAT文件错误：{os.path.basename(image_path)} 中未找到'BW'键")
        except Exception as e:
            raise ValueError(f"读取MAT文件失败：{os.path.basename(image_path)} → {e}")
    else:
        raise ValueError(f"不支持的文件格式：{ext}（仅支持.tif/.tiff/.mat）")
    
    img = img.astype(np.float32)
    if np.max(img) > 1:
        img = np.where(img > 127, 1, 0)
    else:
        img = np.where(img > 0.5, 1, 0)
    
    if img.shape != (96, 96, 96):
        raise ValueError(f"图像尺寸错误：预期96×96×96，当前为{img.shape}")
    
    phi = np.mean(img)
    tpf, r_values = calculate_tpf(img, phi)
    lpf, l_values = calculate_lpf(img)
    
    return tpf, r_values, lpf, l_values


# ------------------- 批量处理函数（支持样本限制） ------------------- #
def process_dataset(
    folder_path, 
    include_key=None,  
    exclude_keys=None, 
    case_sensitive=False,
    sample_limit=None  # 新增：单文件夹样本限制
):
    all_tpf = []     # 当前文件夹的TPF结果（最多sample_limit个）
    all_lpf = []     # 当前文件夹的LPF结果（最多sample_limit个）
    all_r = None     # 当前文件夹的r_values（所有图像尺寸一致）
    all_l = None     # 当前文件夹的l_values（取最长序列）
    collected = 0    # 已收集的样本数（新增）
    
    # 获取文件夹内所有支持的图像文件（大小写不敏感）
    image_files = []
    image_files.extend(glob.glob(os.path.join(folder_path, '*.[Tt][Ii][Ff]')))    
    image_files.extend(glob.glob(os.path.join(folder_path, '*.[Tt][Ii][Ff][Ff]'))) 
    image_files.extend(glob.glob(os.path.join(folder_path, '*.[Mm][Aa][Tt]')))     
    image_files = list(set(image_files))  # 去重
    
    if not image_files:
        print(f"警告：文件夹 {os.path.basename(folder_path)} 中未找到支持的图像文件")
        return all_tpf, all_r, all_lpf, all_l
    
    # 处理排除关键字（转为小写，若不区分大小写）
    if exclude_keys is None:
        exclude_keys = []
    if not case_sensitive:
        exclude_keys = [key.lower() for key in exclude_keys]
    
    # 遍历文件（带进度条）
    for file in tqdm(image_files, desc=f"处理文件夹 {os.path.basename(folder_path)}"):
        filename = os.path.basename(file)
        
        # 1. 筛选包含关键字的文件
        if include_key is not None:
            if not case_sensitive:
                filename_low = filename.lower()
                include_key_low = include_key.lower()
            else:
                filename_low = filename
                include_key_low = include_key
            if include_key_low not in filename_low:
                continue
        
        # 2. 筛选不包含排除关键字的文件
        if exclude_keys:
            if not case_sensitive:
                filename_low = filename.lower()
            else:
                filename_low = filename
            if any(key in filename_low for key in exclude_keys):
                continue
        
        # 3. 处理符合条件的文件（控制样本数量）
        try:
            tpf, r_values, lpf, l_values = calculate_tpf_lpf(file)
            
            # 初始化r_values（所有图像尺寸一致）
            if all_r is None:
                all_r = r_values
            
            # 更新最长l_values（确保后续补零对齐）
            if all_l is None or len(l_values) > len(all_l):
                all_l = l_values
            
            # 添加结果到列表（仅处理成功的样本）
            all_tpf.append(tpf)
            all_lpf.append(lpf)
            collected += 1  # 计数器递增
            
            # 达到样本限制，停止处理当前文件夹的后续文件
            if sample_limit is not None and collected >= sample_limit:
                print(f"已收集 {collected} 个样本（达到限制 {sample_limit}），停止处理当前文件夹")
                break
        
        except ValueError as e:
            print(f"警告：文件 {filename} 处理失败（跳过）→ {e}")
        except Exception as e:
            print(f"警告：文件 {filename} 发生未知错误（跳过）→ {e}")
    
    # LPF长度补零（统一到当前文件夹的最长l_values）
    if all_l is not None and len(all_lpf) > 0:
        max_l_len = len(all_l)
        for i in range(len(all_lpf)):
            if len(all_lpf[i]) < max_l_len:
                all_lpf[i] = np.pad(all_lpf[i], (0, max_l_len - len(all_lpf[i])), 'constant')
    
    return all_tpf, all_r, all_lpf, all_l


# ------------------- 统计分析函数（不变） ------------------- #
def compute_stats(data_list, x_values):
    if not data_list:
        return (np.array([]), np.array([]), np.array([]), np.array([]), x_values)
    
    data_array = np.array(data_list)
    mean = np.mean(data_array, axis=0)
    std = np.std(data_array, axis=0)
    min_val = np.min(data_array, axis=0)
    max_val = np.max(data_array, axis=0)
    
    assert len(x_values) == len(mean), "x轴与数据长度不匹配"
    return mean, std, min_val, max_val, x_values


# ------------------- 主流程（修正样本限制逻辑） ------------------- #
if __name__ == "__main__":
    # ------------------- 1. 配置参数（替换为实际值！） ------------------- #
    print('# ------------------- 1. 配置参数 ------------------- #')
    generated_parent = "Ckpt_Slice_ldm/Dim_ckpt_SUNet_try2_2nd_ep"  
    train_folder = "DATA/VAE_DATA/AIMAX_all/mat_cls5_train"         
    gen_include_key = "SUNet"                    
    gen_exclude_keys = ["#1", "#2", "#3"]      
    train_include_key = "class5"                     
    train_exclude_keys = ["#4", "#5", "#6"]    
    gen_folder_pattern = "fake_set_seed*"       
    case_sensitive = False 
    save_file = "stats_params.yaml"  
    gen_sample_limit = 10000  # 生成集总样本限制（取前20个）
    train_sample_limit = 10000  # 训练集样本限制（取前20个）

    # ------------------- 2. 处理生成数据集（控制总样本数） ------------------- #
    print('# ------------------- 2. 处理生成数据集 ------------------- #')
    generated_folders = glob.glob(os.path.join(generated_parent, gen_folder_pattern))
    if not generated_folders:
        raise ValueError(f"未找到符合模式 {gen_folder_pattern} 的生成集子文件夹")

    gen_tpf_list = []  # 总TPF结果（最多gen_sample_limit个）
    gen_lpf_list = []  # 总LPF结果（最多gen_sample_limit个）
    gen_r = None       # 生成集统一的r_values
    gen_l = None       # 生成集统一的l_values
    collected_gen = 0  # 已收集的生成集样本数

    for folder in tqdm(generated_folders, desc="处理生成数据集子文件夹"):
        if gen_sample_limit is not None and collected_gen >= gen_sample_limit:
            print("生成集样本数已达到限制，停止处理后续子文件夹")
            break
        
        remaining_limit = gen_sample_limit - collected_gen if gen_sample_limit is not None else None
        
        try:
            folder_tpf, folder_r, folder_lpf, folder_l = process_dataset(
                folder_path=folder,
                include_key=gen_include_key,
                exclude_keys=gen_exclude_keys,
                case_sensitive=case_sensitive,
                sample_limit=remaining_limit  # 传递剩余样本限制
            )
            
            gen_tpf_list.extend(folder_tpf)
            gen_lpf_list.extend(folder_lpf)
            collected_gen += len(folder_tpf)
            
            if gen_r is None:
                gen_r = folder_r
            if gen_l is None or len(folder_l) > len(gen_l):
                gen_l = folder_l
        
        except Exception as e:
            print(f"警告：子文件夹 {os.path.basename(folder)} 处理失败（跳过）→ {e}")

    if not gen_tpf_list or not gen_lpf_list:
        raise ValueError("生成数据集没有有效的TPF/LPF结果")

    # LPF补零（统一到生成集的最长l_values）
    max_l_len = len(gen_l)
    for i in range(len(gen_lpf_list)):
        if len(gen_lpf_list[i]) < max_l_len:
            gen_lpf_list[i] = np.pad(gen_lpf_list[i], (0, max_l_len - len(gen_lpf_list[i])), 'constant')

    # ------------------- 3. 处理训练数据集（控制样本数） ------------------- #
    print('# ------------------- 3. 处理训练数据集 ------------------- #')
    try:
        train_tpf_list, train_r, train_lpf_list, train_l = process_dataset(
            folder_path=train_folder,
            include_key=train_include_key,
            exclude_keys=train_exclude_keys,
            case_sensitive=case_sensitive,
            sample_limit=train_sample_limit  # 传递训练集样本限制
        )
    except Exception as e:
        raise ValueError(f"训练数据集处理失败 → {e}")

    if not train_tpf_list or not train_lpf_list:
        raise ValueError("训练数据集没有有效的TPF/LPF结果")

    # ------------------- 4. 计算统计量（不变） ------------------- #
    print('# ------------------- 4. 计算统计量 ------------------- #')
    # TPF统计
    gen_tpf_mean, gen_tpf_std, gen_tpf_min, gen_tpf_max, r_values = compute_stats(gen_tpf_list, gen_r)
    train_tpf_mean, train_tpf_std, train_tpf_min, train_tpf_max, _ = compute_stats(train_tpf_list, train_r)

    # LPF统计
    gen_lpf_mean, gen_lpf_std, gen_lpf_min, gen_lpf_max, l_values = compute_stats(gen_lpf_list, gen_l)
    train_lpf_mean, train_lpf_std, train_lpf_min, train_lpf_max, _ = compute_stats(train_lpf_list, train_l)

    # ------------------- 5. 保存参数与统计值（不变） ------------------- #
    print('# ------------------- 5. 保存参数与统计值 ------------------- #')
    params_dict = {
        "generated_parent": generated_parent,
        "train_folder": train_folder,
        "gen_include_key": gen_include_key,
        "gen_exclude_keys": gen_exclude_keys,
        "train_include_key": train_include_key,
        "train_exclude_keys": train_exclude_keys,
        "gen_folder_pattern": gen_folder_pattern,
        "case_sensitive": case_sensitive,
        "gen_sample_limit": gen_sample_limit,  # 新增：生成集样本限制
        "train_sample_limit": train_sample_limit,  # 新增：训练集样本限制
        "save_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    stats_dict = {
        "tpf": {
            "generated": {
                "mean": gen_tpf_mean.tolist(),
                "std": gen_tpf_std.tolist(),
                "min": gen_tpf_min.tolist(),
                "max": gen_tpf_max.tolist(),
                "r_values": r_values.tolist()
            },
            "train": {
                "mean": train_tpf_mean.tolist(),
                "std": train_tpf_std.tolist(),
                "min": train_tpf_min.tolist(),
                "max": train_tpf_max.tolist(),
                "r_values": train_r.tolist()
            }
        },
        "lpf": {
            "generated": {
                "mean": gen_lpf_mean.tolist(),
                "std": gen_lpf_std.tolist(),
                "min": gen_lpf_min.tolist(),
                "max": gen_lpf_max.tolist(),
                "l_values": l_values.tolist()
            },
            "train": {
                "mean": train_lpf_mean.tolist(),
                "std": train_lpf_std.tolist(),
                "min": train_lpf_min.tolist(),
                "max": train_lpf_max.tolist(),
                "l_values": train_l.tolist()
            }
        }
    }

    data_to_save = {"parameters": params_dict, "statistics": stats_dict}

    try:
        os.makedirs(os.path.dirname(save_file) or ".", exist_ok=True)
        with open(save_file, "w", encoding="utf-8") as f:
            yaml.dump(data_to_save, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        print(f"✅ 统计值与参数已保存到：{os.path.abspath(save_file)}")
    except Exception as e:
        print(f"❌ 保存失败：{str(e)}")

    # ------------------- 6. 绘制TPF对比图（不变） ------------------- #
    print('# ------------------- 6. 绘制TPF对比图 ------------------- #')
    plt.figure(figsize=(12, 6))
    plt.plot(r_values, train_tpf_mean, label="Origin (Mean)", color="#1E88E5", linewidth=2)
    plt.fill_between(r_values, train_tpf_mean - train_tpf_std, train_tpf_mean + train_tpf_std, color="#BBDEFB", alpha=0.8, label="Origin (Envelop: Mean±Std)")
    plt.plot(r_values, gen_tpf_mean, label="Generated (Mean)", color="#E53935", linewidth=2)
    plt.fill_between(r_values, gen_tpf_mean - gen_tpf_std, gen_tpf_mean + gen_tpf_std, color="#FFCDD2", alpha=0.8, label="Generated (Envelop: Mean±Std)")
    plt.xlabel("Distance", fontsize=12)
    plt.ylabel("$S_2(r)$", fontsize=12)
    plt.title("Two-Point Correlation Function Comparison", fontsize=14, pad=15)
    plt.legend(fontsize=10, loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("tpf_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ------------------- 7. 绘制LPF对比图（不变） ------------------- #
    print('# ------------------- 7. 绘制LPF对比图 ------------------- #')
    plt.figure(figsize=(12, 6))
    plt.plot(l_values, train_lpf_mean, label="Origin (Mean)", color="#1E88E5", linewidth=2)
    plt.fill_between(l_values, train_lpf_mean - train_lpf_std, train_lpf_mean + train_lpf_std, color="#BBDEFB", alpha=0.8, label="Origin (Envelop: Mean±Std)")
    plt.plot(l_values, gen_lpf_mean, label="Generated (Mean)", color="#E53935", linewidth=2)
    plt.fill_between(l_values, gen_lpf_mean - gen_lpf_std, gen_lpf_mean + gen_lpf_std, color="#FFCDD2", alpha=0.8, label="Generated (Envelop: Mean±Std)")
    plt.xlabel("Path Length", fontsize=12)
    plt.ylabel("LPF Probability", fontsize=12)
    plt.title("Linear Path Function Comparison", fontsize=14, pad=15)
    plt.legend(fontsize=10, loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("lpf_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()
    
# (cd ./Ckpt_U_FFT_2nd/ckpt_LDM_NN_cfg_020619_f_SR_FFT_alpha05_ep_twice && zip -r fake_set.zip fake_set)
# (cd ./Ckpt_U_FFT_2nd/ckpt_LDM_NN_cfg_020619_f_SR_FFT_alpha05_ep_twice/fake_set_eval && unzip fake_set_seed0.zip -d fake_set_seed0)

# 3154个生成样本以及1100个训练样本