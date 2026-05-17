# import os
# import glob
# import numpy as np
# import tifffile
# import scipy.io as sio
# from scipy import fft
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import yaml
# from datetime import datetime
# from typing import Dict, List

# # ------------------- 基础计算函数 ------------------- #
# def radial_average(autocorr, shape):
#     z, y, x = np.indices(shape)
#     center = np.array(shape) // 2
#     r = np.sqrt((x - center[2])**2 + (y - center[1])**2 + (z - center[0])**2)
#     r = r.astype(int)
    
#     r_flat = r.ravel()
#     autocorr_flat = autocorr.ravel()
    
#     r_max = min(shape) // 2
#     r_values = np.arange(0, r_max)
#     tpf = np.zeros(r_max)
    
#     for radius in range(r_max):
#         mask = (r_flat == radius)
#         if np.any(mask):
#             tpf[radius] = np.mean(autocorr_flat[mask])
#     return tpf, r_values


# def calculate_tpf(img, phi):
#     padded_shape = tuple(2 * dim for dim in img.shape)
#     padded_img = np.zeros(padded_shape, dtype=img.dtype)
#     slices = tuple(slice(0, dim) for dim in img.shape)
#     padded_img[slices] = img
    
#     F = fft.fftn(padded_img)
#     autocorr = fft.ifftn(F * np.conj(F)).real
#     autocorr = autocorr / img.size  # 原始S₂(r)（未归一化）
    
#     autocorr = fft.fftshift(autocorr)  # 平移原点到中心
#     center_slices = tuple(slice(dim//2, dim + dim//2) for dim in img.shape)
#     autocorr = autocorr[center_slices]
    
#     tpf, r_values = radial_average(autocorr, img.shape)
#     return tpf, r_values


# def calculate_lpf(img):
#     directions = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
#     all_lengths = []
    
#     for dx, dy, dz in directions:
#         lengths = np.zeros_like(img, dtype=np.uint8)
#         z_range = range(img.shape[0]-1, -1, -1) if dz < 0 else range(img.shape[0])
#         y_range = range(img.shape[1]-1, -1, -1) if dy < 0 else range(img.shape[1])
#         x_range = range(img.shape[2]-1, -1, -1) if dx < 0 else range(img.shape[2])
        
#         for z in z_range:
#             for y in y_range:
#                 for x in x_range:
#                     if img[z, y, x] == 0:
#                         lengths[z, y, x] = 0
#                         continue
#                     prev_z, prev_y, prev_x = z - dz, y - dy, x - dx
#                     if (0 <= prev_z < img.shape[0] and 
#                         0 <= prev_y < img.shape[1] and 
#                         0 <= prev_x < img.shape[2]):
#                         lengths[z, y, x] = lengths[prev_z, prev_y, prev_x] + 1
#                     else:
#                         lengths[z, y, x] = 1
#         all_lengths.extend(lengths[img == 1].ravel())
    
#     if not all_lengths:
#         return np.array([]), np.array([])
#     max_length = max(all_lengths)
#     l_values = np.arange(1, max_length + 1)
#     total_paths = len(all_lengths)
#     lpf = np.zeros(max_length + 1)
#     for L in l_values:
#         lpf[L] = np.sum(np.array(all_lengths) >= L) / total_paths
#     return lpf[1:], l_values


# def calculate_tpf_lpf(image_path):
#     """计算单张图像的TPF、LPF和孔隙度"""
#     ext = os.path.splitext(image_path)[1].lower()
    
#     if ext in ('.tif', '.tiff'):
#         img = tifffile.imread(image_path)
#     elif ext == '.mat':
#         try:
#             mat_data = sio.loadmat(image_path)
#             img = mat_data['BW']
#         except KeyError:
#             raise ValueError(f"MAT文件错误：{os.path.basename(image_path)} 中未找到'BW'键")
#         except Exception as e:
#             raise ValueError(f"读取MAT文件失败：{os.path.basename(image_path)} → {e}")
#     else:
#         raise ValueError(f"不支持的文件格式：{ext}（仅支持.tif/.tiff/.mat）")
    
#     # 二值化（确保0-1）
#     img = img.astype(np.float32)
#     if np.max(img) > 1:
#         img = np.where(img > 127, 1, 0)
#     else:
#         img = np.where(img > 0.5, 1, 0)
    
#     # 检查尺寸（必须96×96×96）
#     if img.shape != (96, 96, 96):
#         raise ValueError(f"图像尺寸错误：预期96×96×96，当前为{img.shape}")
    
#     # 计算孔隙度（φ）
#     phi = np.mean(img)
#     # 计算TPF和LPF
#     tpf, r_values = calculate_tpf(img, phi)
#     lpf, l_values = calculate_lpf(img)
    
#     return tpf, r_values, lpf, l_values, phi  # 新增返回孔隙度


# # ------------------- 批量处理函数（支持样本限制与孔隙度收集） ------------------- #
# def process_dataset(
#     folder_path, 
#     include_key=None,  
#     exclude_keys=None, 
#     case_sensitive=False,
#     sample_limit=None  
# ):
#     """
#     处理单个文件夹中的图像，返回TPF、LPF、孔隙度及对应的x轴值
#     返回：all_tpf（列表）、all_r（数组）、all_lpf（列表）、all_l（数组）、all_phi（列表）
#     """
#     all_tpf = []     # 当前文件夹的TPF结果（每个元素是1D数组）
#     all_lpf = []     # 当前文件夹的LPF结果（每个元素是1D数组）
#     all_phi = []     # 当前文件夹的孔隙度结果（每个元素是浮点数）
#     all_r = None     # 当前文件夹的r_values（统一，1D数组）
#     all_l = None     # 当前文件夹的l_values（统一，1D数组）
#     collected = 0    # 已收集的样本数（控制样本限制）
    
#     # 获取文件夹内所有支持的图像文件（大小写不敏感）
#     image_files = []
#     image_files.extend(glob.glob(os.path.join(folder_path, '*.[Tt][Ii][Ff]')))    
#     image_files.extend(glob.glob(os.path.join(folder_path, '*.[Tt][Ii][Ff][Ff]'))) 
#     image_files.extend(glob.glob(os.path.join(folder_path, '*.[Mm][Aa][Tt]')))     
#     image_files = list(set(image_files))  # 去重
    
#     if not image_files:
#         print(f"警告：文件夹 {os.path.basename(folder_path)} 中未找到支持的图像文件")
#         return all_tpf, all_r, all_lpf, all_l, all_phi
    
#     # 处理排除关键字（转为小写，若不区分大小写）
#     if exclude_keys is None:
#         exclude_keys = []
#     if not case_sensitive:
#         exclude_keys = [key.lower() for key in exclude_keys]
    
#     # 遍历文件（带进度条）
#     for file in tqdm(image_files, desc=f"处理文件夹 {os.path.basename(folder_path)}"):
#         filename = os.path.basename(file)
        
#         # 1. 筛选包含关键字的文件
#         if include_key is not None:
#             if not case_sensitive:
#                 filename_low = filename.lower()
#                 include_key_low = include_key.lower()
#             else:
#                 filename_low = filename
#                 include_key_low = include_key
#             if include_key_low not in filename_low:
#                 continue
        
#         # 2. 筛选不包含排除关键字的文件
#         if exclude_keys:
#             if not case_sensitive:
#                 filename_low = filename.lower()
#             else:
#                 filename_low = filename
#             if any(key in filename_low for key in exclude_keys):
#                 continue
        
#         # 3. 处理符合条件的文件（控制样本数量）
#         try:
#             tpf, r_values, lpf, l_values, phi = calculate_tpf_lpf(file)
            
#             # 初始化r_values（所有图像尺寸一致，取第一个有效结果）
#             if all_r is None:
#                 all_r = r_values
            
#             # 更新最长l_values（确保后续补零对齐）
#             if all_l is None or len(l_values) > len(all_l):
#                 all_l = l_values
            
#             # 添加结果到列表（仅处理成功的样本）
#             all_tpf.append(tpf)
#             all_lpf.append(lpf)
#             all_phi.append(phi)  # 收集孔隙度
#             collected += 1  # 计数器递增
            
#             # 达到样本限制，停止处理当前文件夹的后续文件
#             if sample_limit is not None and collected >= sample_limit:
#                 print(f"已收集 {collected} 个样本（达到限制 {sample_limit}），停止处理当前文件夹")
#                 break
        
#         except ValueError as e:
#             print(f"警告：文件 {filename} 处理失败（跳过）→ {e}")
#         except Exception as e:
#             print(f"警告：文件 {filename} 发生未知错误（跳过）→ {e}")
    
#     # LPF长度补零（统一到当前文件夹的最长l_values）
#     if all_l is not None and len(all_lpf) > 0:
#         max_l_len = len(all_l)
#         for i in range(len(all_lpf)):
#             if len(all_lpf[i]) < max_l_len:
#                 all_lpf[i] = np.pad(all_lpf[i], (0, max_l_len - len(all_lpf[i])), 'constant')
    
#     return all_tpf, all_r, all_lpf, all_l, all_phi


# # ------------------- 统计分析函数 ------------------- #
# def compute_stats(data_list, x_values):
#     """计算数据列表的统计值（均值、标准差、最小值、最大值）"""
#     if not data_list:
#         return (np.array([]), np.array([]), np.array([]), np.array([]), x_values)
    
#     data_array = np.array(data_list)
#     mean = np.mean(data_array, axis=0)
#     std = np.std(data_array, axis=0)
#     min_val = np.min(data_array, axis=0)
#     max_val = np.max(data_array, axis=0)
    
#     assert len(x_values) == len(mean), "x轴与数据长度不匹配"
#     return mean, std, min_val, max_val, x_values


# def compute_phi_stats(phi_list):
#     """计算孔隙度的统计值（均值、标准差、四分位数等）"""
#     if not phi_list:
#         return None
#     phi_array = np.array(phi_list)
#     return {
#         "mean": float(np.mean(phi_array)),
#         "std": float(np.std(phi_array)),
#         "min": float(np.min(phi_array)),
#         "max": float(np.max(phi_array)),
#         "25%": float(np.percentile(phi_array, 25)),  # 下四分位数
#         "50%": float(np.percentile(phi_array, 50)),  # 中位数
#         "75%": float(np.percentile(phi_array, 75))   # 上四分位数
#     }


# # ------------------- 归一化函数 ------------------- #
# def normalize_tpf(tpf_list, phi_list):
#     """将TPF归一化（tpf/phi），返回归一化后的TPF列表"""
#     if len(tpf_list) != len(phi_list):
#         raise ValueError("TPF列表与孔隙度列表长度不一致")
    
#     norm_tpf_list = []
#     for tpf, phi in zip(tpf_list, phi_list):
#         if phi == 0:
#             raise ValueError("孔隙度为0，无法归一化")
#         norm_tpf = tpf / phi  # 归一化：S₂(r)/φ
#         norm_tpf_list.append(norm_tpf)
    
#     return norm_tpf_list


# # ------------------- 数据读取函数 ------------------- #

# def load_yaml_data(yaml_path: str) -> Dict:
#     """加载YAML文件数据"""
#     with open(yaml_path, 'r', encoding='utf-8') as f:
#         data = yaml.load(f, Loader=yaml.FullLoader)
#     return data


# def extract_porosity_data(data: Dict) -> List[np.ndarray]:
#     """提取孔隙度原始数据（用于箱式图）"""
#     gen_phi = np.array(data['parameters'].get('gen_phi_list', []))
#     train_phi = np.array(data['parameters'].get('train_phi_list', []))
#     if len(gen_phi) == 0 or len(train_phi) == 0:
#         raise ValueError("YAML文件中未找到孔隙度原始数据（gen_phi_list/train_phi_list）")
#     return gen_phi, train_phi


# def extract_tpf_data(data: Dict, norm: bool = False) -> Dict:
#     """提取TPF统计数据（原始/归一化）"""
#     key = 'norm_tpf' if norm else 'tpf'
#     stats = data['statistics'][key]
    
#     # 生成集数据
#     gen = {
#         'mean': np.array(stats['generated']['mean']),
#         'std': np.array(stats['generated']['std']),
#         'r': np.array(stats['generated']['r_values'])
#     }
#     # 训练集数据
#     train = {
#         'mean': np.array(stats['train']['mean']),
#         'std': np.array(stats['train']['std']),
#         'r': np.array(stats['train']['r_values'])
#     }
    
#     # 验证距离值一致性（确保生成集与训练集的r轴一致）
#     if not np.array_equal(gen['r'], train['r']):
#         raise ValueError(f"{'归一化' if norm else '原始'}TPF的生成集与训练集r轴不一致")
    
#     return {'gen': gen, 'train': train}


# def extract_lpf_data(data: Dict) -> Dict:
#     """提取LPF统计数据"""
#     stats = data['statistics']['lpf']
    
#     # 生成集数据
#     gen = {
#         'mean': np.array(stats['generated']['mean']),
#         'std': np.array(stats['generated']['std']),
#         'l': np.array(stats['generated']['l_values'])
#     }
#     # 训练集数据
#     train = {
#         'mean': np.array(stats['train']['mean']),
#         'std': np.array(stats['train']['std']),
#         'l': np.array(stats['train']['l_values'])
#     }
    
#     # 验证路径长度值一致性
#     if not np.array_equal(gen['l'], train['l']):
#         raise ValueError("LPF的生成集与训练集路径长度轴不一致")
    
#     return {'gen': gen, 'train': train}


# # ------------------- 画图函数 ------------------- #
# def plot_porosity_boxplot(gen_phi: np.ndarray, train_phi: np.ndarray, save_path: str):
#     """绘制孔隙度箱式图"""
#     print('\n# ------------------- 绘制孔隙度箱式图 ------------------- #')
#     plt.figure(figsize=(8, 6))
    
#     # 数据与标签
#     data = [train_phi, gen_phi]
#     labels = ['Train', 'Generated']
#     colors = ['#1E88E5', '#E53935']  # 与原代码保持一致的颜色
    
#     # 绘制箱式图（显示异常值）
#     boxplot = plt.boxplot(
#         data,
#         labels=labels,
#         patch_artist=True,  # 允许填充颜色
#         showfliers=True,    # 显示异常值
#         widths=0.5,         # 箱体宽度
#         medianprops={"color": "white", "linewidth": 2},  # 中位数线样式
#         whiskerprops={"color": "black", "linewidth": 1.5},  #  whiskers样式
#         capprops={"color": "black", "linewidth": 1.5}  #  caps样式
#     )
    
#     # 设置箱体颜色
#     for patch, color in zip(boxplot['boxes'], colors):
#         patch.set_facecolor(color)
#         patch.set_edgecolor('black')  # 箱体边框颜色
    
#     # 设置异常值样式（红色圆点，半透明）
#     for flier in boxplot['fliers']:
#         flier.set(marker='o', color='red', alpha=0.5)
    
#     # 添加标签与标题
#     plt.ylabel('Porosity ($\phi$)', fontsize=12)
#     plt.title('Porosity Distribution Comparison', fontsize=14, pad=15)
#     plt.grid(True, linestyle="--", alpha=0.7)  # 网格线
#     plt.tight_layout()  # 自动调整布局
    
#     # 保存图片
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     print(f"✅ 孔隙度箱式图已保存到：{save_path}")
#     plt.show()


# def plot_tpf_comparison(tpf_data: Dict, save_path: str, norm: bool = False):
#     """绘制TPF对比图（原始/归一化）"""
#     if norm:
#         print('\n# ------------------- 绘制归一化TPF对比图 ------------------- #')
#     else:
#         print('\n# ------------------- 绘制原始TPF对比图 ------------------- #')
#     plt.figure(figsize=(12, 6))
    
#     # 训练集：均值±标准差
#     plt.plot(
#         tpf_data['train']['r'],
#         tpf_data['train']['mean'],
#         label="Train (Mean)",
#         color="#1E88E5",
#         linewidth=2
#     )
#     plt.fill_between(
#         tpf_data['train']['r'],
#         tpf_data['train']['mean'] - tpf_data['train']['std'],
#         tpf_data['train']['mean'] + tpf_data['train']['std'],
#         color="#BBDEFB",
#         alpha=0.8,
#         label="Train (Mean±Std)"
#     )
    
#     # 生成集：均值±标准差
#     plt.plot(
#         tpf_data['gen']['r'],
#         tpf_data['gen']['mean'],
#         label="Generated (Mean)",
#         color="#E53935",
#         linewidth=2
#     )
#     plt.fill_between(
#         tpf_data['gen']['r'],
#         tpf_data['gen']['mean'] - tpf_data['gen']['std'],
#         tpf_data['gen']['mean'] + tpf_data['gen']['std'],
#         color="#FFCDD2",
#         alpha=0.8,
#         label="Generated (Mean±Std)"
#     )
    
#     # 设置标签与标题
#     plt.xlabel("Distance", fontsize=12)
#     plt.ylabel("$S_2(r)/\phi$" if norm else "$S_2(r)$", fontsize=12)
#     plt.title(
#         "Normalized Two-Point Correlation Function Comparison" if norm 
#         else "Original Two-Point Correlation Function Comparison",
#         fontsize=14,
#         pad=15
#     )
#     plt.legend(fontsize=10, loc="upper right")  # 图例位置（右上角）
#     plt.grid(True, linestyle="--", alpha=0.7)  # 网格线
#     plt.tight_layout()  # 自动调整布局
    
#     # 保存图片
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     print(f"✅ {'归一化' if norm else '原始'}TPF对比图已保存到：{save_path}")
#     plt.show()


# def plot_lpf_comparison(lpf_data: Dict, save_path: str):
#     """绘制LPF对比图"""
#     print('\n# ------------------- 绘制LPF对比图 ------------------- #')
#     plt.figure(figsize=(12, 6))
    
#     # 训练集：均值±标准差
#     plt.plot(
#         lpf_data['train']['l'],
#         lpf_data['train']['mean'],
#         label="Train (Mean)",
#         color="#1E88E5",
#         linewidth=2
#     )
#     plt.fill_between(
#         lpf_data['train']['l'],
#         lpf_data['train']['mean'] - lpf_data['train']['std'],
#         lpf_data['train']['mean'] + lpf_data['train']['std'],
#         color="#BBDEFB",
#         alpha=0.8,
#         label="Train (Mean±Std)"
#     )
    
#     # 生成集：均值±标准差
#     plt.plot(
#         lpf_data['gen']['l'],
#         lpf_data['gen']['mean'],
#         label="Generated (Mean)",
#         color="#E53935",
#         linewidth=2
#     )
#     plt.fill_between(
#         lpf_data['gen']['l'],
#         lpf_data['gen']['mean'] - lpf_data['gen']['std'],
#         lpf_data['gen']['mean'] + lpf_data['gen']['std'],
#         color="#FFCDD2",
#         alpha=0.8,
#         label="Generated (Mean±Std)"
#     )
    
#     # 设置标签与标题
#     plt.xlabel("Path Length", fontsize=12)
#     plt.ylabel("LPF Probability", fontsize=12)
#     plt.title("Linear Path Function Comparison", fontsize=14, pad=15)
#     plt.legend(fontsize=10, loc="upper right")  # 图例位置（右上角）
#     plt.grid(True, linestyle="--", alpha=0.7)  # 网格线
#     plt.tight_layout()  # 自动调整布局
    
#     # 保存图片
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     print(f"✅ LPF对比图已保存到：{save_path}")
#     plt.show()



# def cmpt_slbox(generated_parent, train_folder, gen_include_key, gen_exclude_keys, train_include_key, train_exclude_keys, gen_folder_pattern,save_dir,
#                gen_sample_limit=10000, train_sample_limit = 10000, case_sensitive=False):
#     '''
#     generated_parent = "Ckpt_U_FFT_2nd/ckpt_LDM_NN_cfg_020619_f_SR_FFT_alpha05_ep_twice/fake_set_eval"  
#     # 训练数据集：单个文件夹（如原始数据集）
#     train_folder = "data/to_resize_96_class_allin"         
#     # 生成集：必须包含的关键字（文件名含此字符串才会被处理）
#     gen_include_key = "cls0"                    
#     # 生成集：必须排除的关键字列表（文件名含任何一个均跳过）
#     gen_exclude_keys = ["#1", "#2", "#3"]      
#     # 训练集：必须包含的关键字（文件名含此字符串才会被处理）
#     train_include_key = "class1"                     
#     # 训练集：必须排除的关键字列表（文件名含任何一个均跳过）
#     train_exclude_keys = ["#4", "#5", "#6"]    
#     # 生成集子文件夹命名模式（匹配子文件夹，如"fake_set_seed*"）
#     gen_folder_pattern = "fake_set_seed*"       
#     # 是否区分关键字大小写（默认不区分）
#     case_sensitive = False
#     # 结果保存路径（YAML+图片）
#     save_dir = f"eval_cfg_U_FFT/{gen_include_key}"  
#     # 生成/训练集总样本限制（取前N个样本）
#     gen_sample_limit = 10  
#     train_sample_limit = 10
#     '''
#     # ------------------- 2. 初始化保存目录 ------------------- #
#     print('# ------------------- 2. 初始化保存目录 ------------------- #')
#     os.makedirs(save_dir, exist_ok=True)
#     save_file = os.path.join(save_dir, f"stats_params_{gen_include_key}.yaml")

#     # ------------------- 3. 处理生成数据集（收集TPF、LPF、孔隙度） ------------------- #
#     print('\n# ------------------- 处理生成数据集 ------------------- #')
#     generated_folders = glob.glob(os.path.join(generated_parent, gen_folder_pattern))
#     if not generated_folders:
#         raise ValueError(f"未找到符合模式 {gen_folder_pattern} 的生成集子文件夹")

#     gen_tpf_list = []    # 生成集所有样本的TPF（列表，每个元素是1D数组）
#     gen_lpf_list = []    # 生成集所有样本的LPF（列表，每个元素是1D数组）
#     gen_phi_list = []    # 生成集所有样本的孔隙度（列表，每个元素是浮点数）
#     gen_r = None         # 生成集统一的r_values（1D数组）
#     gen_l = None         # 生成集统一的l_values（1D数组）
#     collected_gen = 0    # 已收集的生成集样本数

#     for folder in tqdm(generated_folders, desc="遍历生成集子文件夹"):
#         if gen_sample_limit is not None and collected_gen >= gen_sample_limit:
#             print("生成集样本数已达到限制，停止处理后续子文件夹")
#             break
        
#         # 计算当前子文件夹可收集的剩余样本数
#         remaining_limit = gen_sample_limit - collected_gen if gen_sample_limit is not None else None
        
#         try:
#             # 处理当前子文件夹（获取TPF、LPF、孔隙度）
#             folder_tpf, folder_r, folder_lpf, folder_l, folder_phi = process_dataset(
#                 folder_path=folder,
#                 include_key=gen_include_key,
#                 exclude_keys=gen_exclude_keys,
#                 case_sensitive=case_sensitive,
#                 sample_limit=remaining_limit
#             )
            
#             # 累加结果到总列表
#             gen_tpf_list.extend(folder_tpf)
#             gen_lpf_list.extend(folder_lpf)
#             gen_phi_list.extend(folder_phi)
#             collected_gen += len(folder_tpf)
            
#             # 更新生成集的r_values和l_values（取第一个有效结果）
#             if gen_r is None:
#                 gen_r = folder_r
#             if gen_l is None or len(folder_l) > len(gen_l):
#                 gen_l = folder_l
        
#         except Exception as e:
#             print(f"警告：子文件夹 {os.path.basename(folder)} 处理失败（跳过）→ {e}")

#     # 验证生成集结果有效性
#     if not gen_tpf_list or not gen_lpf_list or not gen_phi_list:
#         raise ValueError("生成数据集没有有效的TPF/LPF/孔隙度结果")

#     # ------------------- 4. 处理训练数据集（收集TPF、LPF、孔隙度） ------------------- #
#     print('\n# ------------------- 处理训练数据集 ------------------- #')
#     try:
#         # 处理训练集文件夹（获取TPF、LPF、孔隙度）
#         train_tpf_list, train_r, train_lpf_list, train_l, train_phi_list = process_dataset(
#             folder_path=train_folder,
#             include_key=train_include_key,
#             exclude_keys=train_exclude_keys,
#             case_sensitive=case_sensitive,
#             sample_limit=train_sample_limit
#         )
#     except Exception as e:
#         raise ValueError(f"训练数据集处理失败 → {e}")

#     # 验证训练集结果有效性
#     if not train_tpf_list or not train_lpf_list or not train_phi_list:
#         raise ValueError("训练数据集没有有效的TPF/LPF/孔隙度结果")

#     # ------------------- 5. 计算统计值（TPF、LPF、孔隙度） ------------------- #
#     print('\n# ------------------- 计算统计值 ------------------- #')
#     # 1. TPF统计（原始）
#     gen_tpf_mean, gen_tpf_std, gen_tpf_min, gen_tpf_max, r_values = compute_stats(gen_tpf_list, gen_r)
#     train_tpf_mean, train_tpf_std, train_tpf_min, train_tpf_max, _ = compute_stats(train_tpf_list, train_r)

#     # 2. LPF统计（原始）
#     gen_lpf_mean, gen_lpf_std, gen_lpf_min, gen_lpf_max, l_values = compute_stats(gen_lpf_list, gen_l)
#     train_lpf_mean, train_lpf_std, train_lpf_min, train_lpf_max, _ = compute_stats(train_lpf_list, train_l)

#     # 3. 孔隙度统计
#     gen_phi_stats = compute_phi_stats(gen_phi_list)
#     train_phi_stats = compute_phi_stats(train_phi_list)

#     # 4. 归一化TPF统计（S₂(r)/φ）
#     norm_gen_tpf_list = normalize_tpf(gen_tpf_list, gen_phi_list)
#     norm_train_tpf_list = normalize_tpf(train_tpf_list, train_phi_list)
#     norm_gen_tpf_mean, norm_gen_tpf_std, norm_gen_tpf_min, norm_gen_tpf_max, _ = compute_stats(norm_gen_tpf_list, gen_r)
#     norm_train_tpf_mean, norm_train_tpf_std, norm_train_tpf_min, norm_train_tpf_max, _ = compute_stats(norm_train_tpf_list, train_r)

#     # ------------------- 6. 保存参数与统计值到YAML ------------------- #
#     print('\n# ------------------- 保存参数与统计值 ------------------- #')
#     params_dict = {
#         "generated_parent": generated_parent,
#         "train_folder": train_folder,
#         "gen_include_key": gen_include_key,
#         "gen_exclude_keys": gen_exclude_keys,
#         "train_include_key": train_include_key,
#         "train_exclude_keys": train_exclude_keys,
#         "gen_folder_pattern": gen_folder_pattern,
#         "case_sensitive": case_sensitive,
#         "gen_sample_limit": gen_sample_limit,
#         "train_sample_limit": train_sample_limit,
#         "save_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         "gen_phi_stats": gen_phi_stats,  # 生成集孔隙度统计
#         "train_phi_stats": train_phi_stats  # 训练集孔隙度统计
#     }

#     stats_dict = {
#         "tpf": {
#             "generated": {
#                 "mean": gen_tpf_mean.tolist(),
#                 "std": gen_tpf_std.tolist(),
#                 "min": gen_tpf_min.tolist(),
#                 "max": gen_tpf_max.tolist(),
#                 "r_values": r_values.tolist()
#             },
#             "train": {
#                 "mean": train_tpf_mean.tolist(),
#                 "std": train_tpf_std.tolist(),
#                 "min": train_tpf_min.tolist(),
#                 "max": train_tpf_max.tolist(),
#                 "r_values": train_r.tolist()
#             }
#         },
#         "norm_tpf": {  # 新增：归一化TPF统计
#             "generated": {
#                 "mean": norm_gen_tpf_mean.tolist(),
#                 "std": norm_gen_tpf_std.tolist(),
#                 "min": norm_gen_tpf_min.tolist(),
#                 "max": norm_gen_tpf_max.tolist(),
#                 "r_values": r_values.tolist()
#             },
#             "train": {
#                 "mean": norm_train_tpf_mean.tolist(),
#                 "std": norm_train_tpf_std.tolist(),
#                 "min": norm_train_tpf_min.tolist(),
#                 "max": norm_train_tpf_max.tolist(),
#                 "r_values": train_r.tolist()
#             }
#         },
#         "lpf": {
#             "generated": {
#                 "mean": gen_lpf_mean.tolist(),
#                 "std": gen_lpf_std.tolist(),
#                 "min": gen_lpf_min.tolist(),
#                 "max": gen_lpf_max.tolist(),
#                 "l_values": l_values.tolist()
#             },
#             "train": {
#                 "mean": train_lpf_mean.tolist(),
#                 "std": train_lpf_std.tolist(),
#                 "min": train_lpf_min.tolist(),
#                 "max": train_lpf_max.tolist(),
#                 "l_values": train_l.tolist()
#             }
#         }
#     }

#     data_to_save = {"parameters": params_dict, "statistics": stats_dict}

#     try:
#         with open(save_file, "w", encoding="utf-8") as f:
#             yaml.dump(data_to_save, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
#         print(f"✅ 统计值与参数已保存到：{os.path.abspath(save_file)}")
#     except Exception as e:
#         print(f"❌ 保存失败：{str(e)}")
        
#     return save_file


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
from typing import Dict, List

# ------------------- 基础计算函数 ------------------- #
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
    autocorr = autocorr / img.size  # 原始S₂(r)（未归一化）
    
    autocorr = fft.fftshift(autocorr)  # 平移原点到中心
    center_slices = tuple(slice(dim//2, dim + dim//2) for dim in img.shape)
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
    """计算单张图像的TPF、LPF和孔隙度"""
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
    
    # 二值化（确保0-1）
    img = img.astype(np.float32)
    if np.max(img) > 1:
        img = np.where(img > 127, 1, 0)
    else:
        img = np.where(img > 0.5, 1, 0)
    
    # 检查尺寸（必须96×96×96）
    if img.shape != (96, 96, 96):
        raise ValueError(f"图像尺寸错误：预期96×96×96，当前为{img.shape}")
    
    # 计算孔隙度（φ）
    phi = np.mean(img)
    # 计算TPF和LPF
    tpf, r_values = calculate_tpf(img, phi)
    lpf, l_values = calculate_lpf(img)
    
    return tpf, r_values, lpf, l_values, phi  # 新增返回孔隙度


# ------------------- 批量处理函数（支持样本限制与孔隙度收集） ------------------- #
def process_dataset(
    folder_path, 
    include_key=None,  
    exclude_keys=None, 
    case_sensitive=False,
    sample_limit=None  
):
    """
    处理单个文件夹中的图像，返回TPF、LPF、孔隙度及对应的x轴值
    返回：all_tpf（列表）、all_r（数组）、all_lpf（列表）、all_l（数组）、all_phi（列表）
    """
    all_tpf = []     # 当前文件夹的TPF结果（每个元素是1D数组）
    all_lpf = []     # 当前文件夹的LPF结果（每个元素是1D数组）
    all_phi = []     # 当前文件夹的孔隙度结果（每个元素是浮点数）
    all_r = None     # 当前文件夹的r_values（统一，1D数组）
    all_l = None     # 当前文件夹的l_values（统一，1D数组）
    collected = 0    # 已收集的样本数（控制样本限制）
    
    # 获取文件夹内所有支持的图像文件（大小写不敏感）
    image_files = []
    image_files.extend(glob.glob(os.path.join(folder_path, '*.[Tt][Ii][Ff]')))    
    image_files.extend(glob.glob(os.path.join(folder_path, '*.[Tt][Ii][Ff][Ff]'))) 
    image_files.extend(glob.glob(os.path.join(folder_path, '*.[Mm][Aa][Tt]')))     
    image_files = list(set(image_files))  # 去重
    
    if not image_files:
        print(f"警告：文件夹 {os.path.basename(folder_path)} 中未找到支持的图像文件")
        return all_tpf, all_r, all_lpf, all_l, all_phi
    
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
            tpf, r_values, lpf, l_values, phi = calculate_tpf_lpf(file)
            
            # 初始化r_values（所有图像尺寸一致，取第一个有效结果）
            if all_r is None:
                all_r = r_values
            
            # 更新最长l_values（确保后续补零对齐）
            if all_l is None or len(l_values) > len(all_l):
                all_l = l_values
            
            # 添加结果到列表（仅处理成功的样本）
            all_tpf.append(tpf)
            all_lpf.append(lpf)
            all_phi.append(phi)  # 收集孔隙度
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
    
    return all_tpf, all_r, all_lpf, all_l, all_phi


# ------------------- 统计分析函数 ------------------- #
def compute_stats(data_list, x_values):
    """计算数据列表的统计值（均值、标准差、最小值、最大值）"""
    if not data_list:
        return (np.array([]), np.array([]), np.array([]), np.array([]), x_values)
    
    data_array = np.array(data_list)
    mean = np.mean(data_array, axis=0)
    std = np.std(data_array, axis=0)
    min_val = np.min(data_array, axis=0)
    max_val = np.max(data_array, axis=0)
    
    assert len(x_values) == len(mean), "x轴与数据长度不匹配"
    return mean, std, min_val, max_val, x_values


def compute_phi_stats(phi_list):
    """计算孔隙度的统计值（均值、标准差、四分位数等）"""
    if not phi_list:
        return None
    phi_array = np.array(phi_list)
    return {
        "mean": float(np.mean(phi_array)),
        "std": float(np.std(phi_array)),
        "min": float(np.min(phi_array)),
        "max": float(np.max(phi_array)),
        "25%": float(np.percentile(phi_array, 25)),  # 下四分位数
        "50%": float(np.percentile(phi_array, 50)),  # 中位数
        "75%": float(np.percentile(phi_array, 75))   # 上四分位数
    }


# ------------------- 归一化函数 ------------------- #
def normalize_tpf(tpf_list, phi_list):
    """将TPF归一化（tpf/phi），返回归一化后的TPF列表"""
    if len(tpf_list) != len(phi_list):
        raise ValueError("TPF列表与孔隙度列表长度不一致")
    
    norm_tpf_list = []
    for tpf, phi in zip(tpf_list, phi_list):
        if phi == 0:
            raise ValueError("孔隙度为0，无法归一化")
        norm_tpf = tpf / phi  # 归一化：S₂(r)/φ
        norm_tpf_list.append(norm_tpf)
    
    return norm_tpf_list


# ------------------- 数据读取函数 ------------------- #

def load_yaml_data(yaml_path: str) -> Dict:
    """加载YAML文件数据"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def extract_porosity_data(data: Dict) -> tuple:
    """提取孔隙度统计数据（用于箱式图）"""
    # 从YAML的parameters中获取统计字典（键名需与YAML一致）
    gen_phi_stats = data['parameters'].get('gen_phi_stats', {})
    train_phi_stats = data['parameters'].get('train_phi_stats', {})
    
    # 验证统计数据是否包含箱式图必需的键（min/25%/50%/75%/max）
    required_keys = ['min', '25%', '50%', '75%', 'max']
    for stats_dict, name in zip([train_phi_stats, gen_phi_stats], ['Train', 'Generated']):
        missing_keys = [key for key in required_keys if key not in stats_dict]
        if missing_keys:
            raise ValueError(f"{name}孔隙度统计数据缺失必需的键：{missing_keys}")
    
    # 提取统计量，组成 tuple（顺序：min, 25%, 50%, 75%, max）
    train_stats = (
        train_phi_stats['min'],
        train_phi_stats['25%'],
        train_phi_stats['50%'],
        train_phi_stats['75%'],
        train_phi_stats['max']
    )
    gen_stats = (
        gen_phi_stats['min'],
        gen_phi_stats['25%'],
        gen_phi_stats['50%'],
        gen_phi_stats['75%'],
        gen_phi_stats['max']
    )
    
    return train_stats, gen_stats  # 返回训练集、生成集的统计量（顺序对应后续画图的顺序）


def extract_tpf_data(data: Dict, norm: bool = False) -> Dict:
    """提取TPF统计数据（原始/归一化）"""
    key = 'norm_tpf' if norm else 'tpf'
    stats = data['statistics'][key]
    
    # 生成集数据
    gen = {
        'mean': np.array(stats['generated']['mean']),
        'std': np.array(stats['generated']['std']),
        'r': np.array(stats['generated']['r_values'])
    }
    # 训练集数据
    train = {
        'mean': np.array(stats['train']['mean']),
        'std': np.array(stats['train']['std']),
        'r': np.array(stats['train']['r_values'])
    }
    
    # 验证距离值一致性（确保生成集与训练集的r轴一致）
    if not np.array_equal(gen['r'], train['r']):
        raise ValueError(f"{'归一化' if norm else '原始'}TPF的生成集与训练集r轴不一致")
    
    return {'gen': gen, 'train': train}


def extract_lpf_data(data: Dict) -> Dict:
    """提取LPF统计数据"""
    stats = data['statistics']['lpf']
    
    # 生成集数据
    gen = {
        'mean': np.array(stats['generated']['mean']),
        'std': np.array(stats['generated']['std']),
        'l': np.array(stats['generated']['l_values'])
    }
    # 训练集数据
    train = {
        'mean': np.array(stats['train']['mean']),
        'std': np.array(stats['train']['std']),
        'l': np.array(stats['train']['l_values'])
    }
    
    # 验证路径长度值一致性
    if not np.array_equal(gen['l'], train['l']):
        raise ValueError("LPF的生成集与训练集路径长度轴不一致")
    
    return {'gen': gen, 'train': train}


# ------------------- 画图函数 ------------------- #
def plot_porosity_boxplot(train_stats: tuple, gen_stats: tuple, save_path: str):
    """使用统计数据绘制孔隙度箱式图"""
    print('\n# ------------------- 绘制孔隙度箱式图 ------------------- #')
    plt.figure(figsize=(8, 6))
    
    # ------------ 核心参数设置 ------------ #
    stats = [train_stats, gen_stats]  # 箱式图统计数据（顺序：训练集、生成集）
    positions = [1, 2]  # 箱体在x轴的位置（1=训练集，2=生成集）
    labels = ['Train', 'Generated']  # 箱体标签
    colors = ['#1E88E5', '#E53935']  # 箱体颜色（与原代码保持一致）
    
    # ------------ 绘制箱式图 ------------ #
    boxplot = plt.boxplot(
        stats,               # 统计数据（每个元素是min/25%/50%/75%/max的tuple）
        positions=positions, # 箱体位置
        widths=0.5,          # 箱体宽度
        patch_artist=True,   # 允许填充颜色
        showfliers=False,    # 不显示异常值（无原始数据，无法计算）
        medianprops={        # 中位数线样式
            "color": "white", 
            "linewidth": 2
        },
        whiskerprops={       #  whiskers（须线）样式
            "color": "black", 
            "linewidth": 1.5
        },
        capprops={           #  caps（须线端点）样式
            "color": "black", 
            "linewidth": 1.5
        }
    )
    
    # ------------ 美化箱体 ------------ #
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)    # 箱体填充颜色
        patch.set_edgecolor('black')  # 箱体边框颜色
    
    # ------------ 设置坐标轴与标题 ------------ #
    plt.xticks(positions, labels, fontsize=12)  # x轴标签（对应箱体位置）
    plt.ylabel('Porosity ($\phi$)', fontsize=12)  # y轴标签（孔隙度）
    plt.title('Porosity Distribution Comparison', fontsize=14, pad=15)  # 标题
    plt.grid(True, linestyle="--", alpha=0.7, axis='y')  # 只显示y轴网格线（更清晰）
    plt.tight_layout()  # 自动调整布局（避免标签重叠）
    
    # ------------ 保存与显示 ------------ #
    plt.savefig(save_path, dpi=800, bbox_inches="tight")  # 保存图片（高分辨率）
    print(f"✅ 孔隙度箱式图已保存到：{save_path}")
    plt.show()  # 显示图片（非交互环境可注释）

    
def plot_tpf_comparison(tpf_data: Dict, save_path: str, norm: bool = False):
    """绘制TPF对比图（原始/归一化）"""
    if norm:
        print('\n# ------------------- 绘制归一化TPF对比图 ------------------- #')
    else:
        print('\n# ------------------- 绘制原始TPF对比图 ------------------- #')
    plt.figure(figsize=(12, 6))
    
    # 训练集：均值±标准差
    plt.plot(
        tpf_data['train']['r'],
        tpf_data['train']['mean'],
        label="Train (Mean)",
        color="#1E88E5",
        linewidth=2
    )
    plt.fill_between(
        tpf_data['train']['r'],
        tpf_data['train']['mean'] - tpf_data['train']['std'],
        tpf_data['train']['mean'] + tpf_data['train']['std'],
        color="#BBDEFB",
        alpha=0.8,
        label="Train (Mean±Std)"
    )
    
    # 生成集：均值±标准差
    plt.plot(
        tpf_data['gen']['r'],
        tpf_data['gen']['mean'],
        label="Generated (Mean)",
        color="#E53935",
        linewidth=2
    )
    plt.fill_between(
        tpf_data['gen']['r'],
        tpf_data['gen']['mean'] - tpf_data['gen']['std'],
        tpf_data['gen']['mean'] + tpf_data['gen']['std'],
        color="#FFCDD2",
        alpha=0.8,
        label="Generated (Mean±Std)"
    )
    
    # 设置标签与标题
    plt.xlabel("Distance", fontsize=12)
    plt.ylabel("$S_2(r)/\phi$" if norm else "$S_2(r)$", fontsize=12)
    plt.title(
        "Normalized Two-Point Correlation Function Comparison" if norm 
        else "Original Two-Point Correlation Function Comparison",
        fontsize=14,
        pad=15
    )
    plt.legend(fontsize=10, loc="upper right")  # 图例位置（右上角）
    plt.grid(True, linestyle="--", alpha=0.7)  # 网格线
    plt.tight_layout()  # 自动调整布局
    
    # 保存图片
    plt.savefig(save_path, dpi=800, bbox_inches="tight")
    print(f"✅ {'归一化' if norm else '原始'}TPF对比图已保存到：{save_path}")
    plt.show()


def plot_lpf_comparison(lpf_data: Dict, save_path: str):
    """绘制LPF对比图"""
    print('\n# ------------------- 绘制LPF对比图 ------------------- #')
    plt.figure(figsize=(12, 6))
    
    # 训练集：均值±标准差
    plt.plot(
        lpf_data['train']['l'],
        lpf_data['train']['mean'],
        label="Train (Mean)",
        color="#1E88E5",
        linewidth=2
    )
    plt.fill_between(
        lpf_data['train']['l'],
        lpf_data['train']['mean'] - lpf_data['train']['std'],
        lpf_data['train']['mean'] + lpf_data['train']['std'],
        color="#BBDEFB",
        alpha=0.8,
        label="Train (Mean±Std)"
    )
    
    # 生成集：均值±标准差
    plt.plot(
        lpf_data['gen']['l'],
        lpf_data['gen']['mean'],
        label="Generated (Mean)",
        color="#E53935",
        linewidth=2
    )
    plt.fill_between(
        lpf_data['gen']['l'],
        lpf_data['gen']['mean'] - lpf_data['gen']['std'],
        lpf_data['gen']['mean'] + lpf_data['gen']['std'],
        color="#FFCDD2",
        alpha=0.8,
        label="Generated (Mean±Std)"
    )
    
    # 设置标签与标题
    plt.xlabel("Path Length", fontsize=12)
    plt.ylabel("LPF Probability", fontsize=12)
    plt.title("Linear Path Function Comparison", fontsize=14, pad=15)
    plt.legend(fontsize=10, loc="upper right")  # 图例位置（右上角）
    plt.grid(True, linestyle="--", alpha=0.7)  # 网格线
    plt.tight_layout()  # 自动调整布局
    
    # 保存图片
    plt.savefig(save_path, dpi=800, bbox_inches="tight")
    print(f"✅ LPF对比图已保存到：{save_path}")
    plt.show()



def cmpt_slbox(generated_parent, train_folder, gen_include_key, gen_exclude_keys, train_include_key, train_exclude_keys, gen_folder_pattern, save_dir,
               gen_sample_limit=10000, train_sample_limit = 10000, case_sensitive=False):
    '''
    generated_parent = "Ckpt_U_FFT_2nd/ckpt_LDM_NN_cfg_020619_f_SR_FFT_alpha05_ep_twice/fake_set_eval"  
    # 训练数据集：单个文件夹（如原始数据集）
    train_folder = "data/to_resize_96_class_allin"         
    # 生成集：必须包含的关键字（文件名含此字符串才会被处理）
    gen_include_key = "cls0"                    
    # 生成集：必须排除的关键字列表（文件名含任何一个均跳过）
    gen_exclude_keys = ["#1", "#2", "#3"]      
    # 训练集：必须包含的关键字（文件名含此字符串才会被处理）
    train_include_key = "class1"                     
    # 训练集：必须排除的关键字列表（文件名含任何一个均跳过）
    train_exclude_keys = ["#4", "#5", "#6"]    
    # 生成集子文件夹命名模式（匹配子文件夹，如"fake_set_seed*"）
    gen_folder_pattern = "fake_set_seed*"       
    # 是否区分关键字大小写（默认不区分）
    case_sensitive = False
    # 结果保存路径（YAML+图片）
    save_dir = f"eval_cfg_U_FFT/{gen_include_key}"  
    # 生成/训练集总样本限制（取前N个样本）
    gen_sample_limit = 10  
    train_sample_limit = 10
    '''
    # ------------------- 2. 初始化保存目录 ------------------- #
    print('# ------------------- 2. 初始化保存目录 ------------------- #')
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, f"stats_params_{gen_include_key}.yaml")

    # ------------------- 3. 处理生成数据集（收集TPF、LPF、孔隙度） ------------------- #
    print('\n# ------------------- 处理生成数据集 ------------------- #')
    generated_folders = glob.glob(os.path.join(generated_parent, gen_folder_pattern))
    if not generated_folders:
        raise ValueError(f"未找到符合模式 {gen_folder_pattern} 的生成集子文件夹")

    gen_tpf_list = []    # 生成集所有样本的TPF（列表，每个元素是1D数组）
    gen_lpf_list = []    # 生成集所有样本的LPF（列表，每个元素是1D数组）
    gen_phi_list = []    # 生成集所有样本的孔隙度（列表，每个元素是浮点数）
    gen_r = None         # 生成集统一的r_values（1D数组）
    gen_l = None         # 生成集统一的l_values（1D数组）
    collected_gen = 0    # 已收集的生成集样本数

    for folder in tqdm(generated_folders, desc="遍历生成集子文件夹"):
        if gen_sample_limit is not None and collected_gen >= gen_sample_limit:
            print("生成集样本数已达到限制，停止处理后续子文件夹")
            break
        
        # 计算当前子文件夹可收集的剩余样本数
        remaining_limit = gen_sample_limit - collected_gen if gen_sample_limit is not None else None
        
        try:
            # 处理当前子文件夹（获取TPF、LPF、孔隙度）
            folder_tpf, folder_r, folder_lpf, folder_l, folder_phi = process_dataset(
                folder_path=folder,
                include_key=gen_include_key,
                exclude_keys=gen_exclude_keys,
                case_sensitive=case_sensitive,
                sample_limit=remaining_limit
            )
            
            # 累加结果到总列表
            gen_tpf_list.extend(folder_tpf)
            gen_lpf_list.extend(folder_lpf)
            gen_phi_list.extend(folder_phi)
            collected_gen += len(folder_tpf)
            
            # 更新生成集的r_values和l_values（取第一个有效结果）
            if gen_r is None:
                gen_r = folder_r
            if gen_l is None or len(folder_l) > len(gen_l):
                gen_l = folder_l
        
        except Exception as e:
            print(f"警告：子文件夹 {os.path.basename(folder)} 处理失败（跳过）→ {e}")

    # 验证生成集结果有效性
    if not gen_tpf_list or not gen_lpf_list or not gen_phi_list:
        raise ValueError("生成数据集没有有效的TPF/LPF/孔隙度结果")

    # ------------------- 4. 处理训练数据集（收集TPF、LPF、孔隙度） ------------------- #
    print('\n# ------------------- 处理训练数据集 ------------------- #')
    try:
        # 处理训练集文件夹（获取TPF、LPF、孔隙度）
        train_tpf_list, train_r, train_lpf_list, train_l, train_phi_list = process_dataset(
            folder_path=train_folder,
            include_key=train_include_key,
            exclude_keys=train_exclude_keys,
            case_sensitive=case_sensitive,
            sample_limit=train_sample_limit
        )
    except Exception as e:
        raise ValueError(f"训练数据集处理失败 → {e}")

    # 验证训练集结果有效性
    if not train_tpf_list or not train_lpf_list or not train_phi_list:
        raise ValueError("训练数据集没有有效的TPF/LPF/孔隙度结果")

    # ------------------- 5. 计算统计值（TPF、LPF、孔隙度） ------------------- #
    print('\n# ------------------- 计算统计值 ------------------- #')
    # 1. TPF统计（原始）
    gen_tpf_mean, gen_tpf_std, gen_tpf_min, gen_tpf_max, r_values = compute_stats(gen_tpf_list, gen_r)
    train_tpf_mean, train_tpf_std, train_tpf_min, train_tpf_max, _ = compute_stats(train_tpf_list, train_r)

    # 2. LPF统计（原始）
    gen_lpf_mean, gen_lpf_std, gen_lpf_min, gen_lpf_max, l_values = compute_stats(gen_lpf_list, gen_l)
    train_lpf_mean, train_lpf_std, train_lpf_min, train_lpf_max, _ = compute_stats(train_lpf_list, train_l)

    # 3. 孔隙度统计
    gen_phi_stats = compute_phi_stats(gen_phi_list)
    train_phi_stats = compute_phi_stats(train_phi_list)

    # 4. 归一化TPF统计（S₂(r)/φ）
    norm_gen_tpf_list = normalize_tpf(gen_tpf_list, gen_phi_list)
    norm_train_tpf_list = normalize_tpf(train_tpf_list, train_phi_list)
    norm_gen_tpf_mean, norm_gen_tpf_std, norm_gen_tpf_min, norm_gen_tpf_max, _ = compute_stats(norm_gen_tpf_list, gen_r)
    norm_train_tpf_mean, norm_train_tpf_std, norm_train_tpf_min, norm_train_tpf_max, _ = compute_stats(norm_train_tpf_list, train_r)

    # ------------------- 6. 保存参数与统计值到YAML ------------------- #
    print('\n# ------------------- 保存参数与统计值 ------------------- #')
    params_dict = {
        "generated_parent": generated_parent,
        "train_folder": train_folder,
        "gen_include_key": gen_include_key,
        "gen_exclude_keys": gen_exclude_keys,
        "train_include_key": train_include_key,
        "train_exclude_keys": train_exclude_keys,
        "gen_folder_pattern": gen_folder_pattern,
        "case_sensitive": case_sensitive,
        "gen_sample_limit": gen_sample_limit,
        "train_sample_limit": train_sample_limit,
        "save_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "gen_phi_stats": gen_phi_stats,  # 生成集孔隙度统计
        "train_phi_stats": train_phi_stats  # 训练集孔隙度统计
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
        "norm_tpf": {  # 新增：归一化TPF统计
            "generated": {
                "mean": norm_gen_tpf_mean.tolist(),
                "std": norm_gen_tpf_std.tolist(),
                "min": norm_gen_tpf_min.tolist(),
                "max": norm_gen_tpf_max.tolist(),
                "r_values": r_values.tolist()
            },
            "train": {
                "mean": norm_train_tpf_mean.tolist(),
                "std": norm_train_tpf_std.tolist(),
                "min": norm_train_tpf_min.tolist(),
                "max": norm_train_tpf_max.tolist(),
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
        with open(save_file, "w", encoding="utf-8") as f:
            yaml.dump(data_to_save, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        print(f"✅ 统计值与参数已保存到：{os.path.abspath(save_file)}")
    except Exception as e:
        print(f"❌ 保存失败：{str(e)}")
        
    return save_file