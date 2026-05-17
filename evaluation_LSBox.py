from LS_Box import *


# ------------------- 主流程：多文件夹处理+统计+可视化 ------------------- #
if __name__ == "__main__":
    cmpt = 'y'
    # 
    if cmpt=='y':
#         # ------------------- 1. 配置参数（替换为实际值！） ------------------- #
#         print('# ------------------- 1. 配置参数（替换为实际值！） ------------------- #')
#         # 生成数据集：父文件夹（包含多个子文件夹，如fake_set_seed*）
#         generated_parent = "Ckpt_Slice_ldm/Dim_ckpt_SUNet_try2_2nd_ep"  
#         # 训练数据集：单个文件夹（如原始数据集）
#         train_folder = "DATA/VAE_DATA/AIMAX_all/mat_cls5_train"         
#         # 生成集：必须包含的关键字（文件名含此字符串才会被处理）
#         gen_include_key = "SUNet"                    
#         # 生成集：必须排除的关键字列表（文件名含任何一个均跳过）
#         gen_exclude_keys = ["#1", "#2", "#3"]      
#         # 训练集：必须包含的关键字（文件名含此字符串才会被处理）
#         train_include_key = "class5"                     
#         # 训练集：必须排除的关键字列表（文件名含任何一个均跳过）
#         train_exclude_keys = ["#4", "#5", "#6"]    
#         # 生成集子文件夹命名模式（匹配子文件夹，如"fake_set_seed*"）
#         gen_folder_pattern = "fake_set_seed0_*"       
#         # 是否区分关键字大小写（默认不区分）
#         case_sensitive = False
#         # 结果保存路径（YAML+图片）
#         save_dir = f"eval_LS_box/{gen_include_key}_phi_bias"  
#         # 生成/训练集总样本限制（取前N个样本）
#         gen_sample_limit = 10000
#         train_sample_limit = 10000

#  # ------------------- 1. 配置参数（替换为实际值！） ------------------- #
#         print('# ------------------- 1. 配置参数（替换为实际值！） ------------------- #')
#         # 生成数据集：父文件夹（包含多个子文件夹，如fake_set_seed*）
#         generated_parent = "Ckpt_Slice_ldm/Dim_ckpt_SPUNet_try_4nd_ep"  
#         # 训练数据集：单个文件夹（如原始数据集）
#         train_folder = "DATA/VAE_DATA/AIMAX_all/mat_cls5_train"         
#         # 生成集：必须包含的关键字（文件名含此字符串才会被处理）
#         gen_include_key = "SPUNet"                    
#         # 生成集：必须排除的关键字列表（文件名含任何一个均跳过）
#         gen_exclude_keys = ["#1", "#2", "#3"]      
#         # 训练集：必须包含的关键字（文件名含此字符串才会被处理）
#         train_include_key = "class5"                     
#         # 训练集：必须排除的关键字列表（文件名含任何一个均跳过）
#         train_exclude_keys = ["#4", "#5", "#6"]    
#         # 生成集子文件夹命名模式（匹配子文件夹，如"fake_set_seed*"）
#         gen_folder_pattern = "fake_set_seed*"       
#         # 是否区分关键字大小写（默认不区分）
#         case_sensitive = False
#         # 结果保存路径（YAML+图片）
#         save_dir = f"eval_LS_box/{gen_include_key}_phi"  
#         # 生成/训练集总样本限制（取前N个样本）
#         gen_sample_limit = 10000
#         train_sample_limit = 10000

#  # ------------------- 1. 配置参数（替换为实际值！） ------------------- #
#         print('# ------------------- 1. 配置参数（替换为实际值！） ------------------- #')
#         # 生成数据集：父文件夹（包含多个子文件夹，如fake_set_seed*）
#         generated_parent = "Ckpt_Slice_ldm/Dim_ckpt_SPUNet_try_4nd_ep"  
#         # 训练数据集：单个文件夹（如原始数据集）
#         train_folder = "DATA/VAE_DATA/AIMAX_all/mat_cls5_train"         
#         # 生成集：必须包含的关键字（文件名含此字符串才会被处理）
#         gen_include_key = "SPUNet"                    
#         # 生成集：必须排除的关键字列表（文件名含任何一个均跳过）
#         # gen_exclude_keys = ["#1", "#2", "#3"]      
#         gen_exclude_keys = ["raw", "#2", "#3"]      
#         # 训练集：必须包含的关键字（文件名含此字符串才会被处理）
#         train_include_key = "class5"                     
#         # 训练集：必须排除的关键字列表（文件名含任何一个均跳过）
#         train_exclude_keys = ["#4", "#5", "#6"]    
#         # 生成集子文件夹命名模式（匹配子文件夹，如"fake_set_seed*"）
#         gen_folder_pattern = "fake_set_seed_*"       
#         # 是否区分关键字大小写（默认不区分）
#         case_sensitive = False
#         # 结果保存路径（YAML+图片）
#         save_dir = f"eval_LS_box/{gen_include_key}_phi_bias"  
#         # 生成/训练集总样本限制（取前N个样本）
#         gen_sample_limit = 10000
#         train_sample_limit = 10000

 # ------------------- 1. 配置参数（替换为实际值！） ------------------- #
        print('# ------------------- 1. 配置参数（替换为实际值！） ------------------- #')
        # 生成数据集：父文件夹（包含多个子文件夹，如fake_set_seed*）
        generated_parent = "Ckpt_Slice_ldm/Dim_ckpt_SPUNet_try_4nd_ep"  
        # 训练数据集：单个文件夹（如原始数据集）
        train_folder = "DATA/VAE_DATA/AIMAX_all/mat_cls5_train"         
        # 生成集：必须包含的关键字（文件名含此字符串才会被处理）
        gen_include_key = "SPUNet"                    
        # 生成集：必须排除的关键字列表（文件名含任何一个均跳过）
        # gen_exclude_keys = ["#1", "#2", "#3"]      
        gen_exclude_keys = ["raw", "#2", "#3"]      
        # 训练集：必须包含的关键字（文件名含此字符串才会被处理）
        train_include_key = "class5"                     
        # 训练集：必须排除的关键字列表（文件名含任何一个均跳过）
        train_exclude_keys = ["#4", "#5", "#6"]    
        # 生成集子文件夹命名模式（匹配子文件夹，如"fake_set_seed*"）
        gen_folder_pattern = "fake_set_seed_bias_0.02*"       
        # 是否区分关键字大小写（默认不区分）
        case_sensitive = False
        # 结果保存路径（YAML+图片）
        save_dir = f"eval_LS_box/{gen_include_key}_phi_bias_0p020"  
        # 生成/训练集总样本限制（取前N个样本）
        gen_sample_limit = 10000
        train_sample_limit = 10000

        save_file = cmpt_slbox(generated_parent,
                               train_folder, 
                               gen_include_key,
                               gen_exclude_keys,
                               train_include_key,
                               train_exclude_keys,
                               gen_folder_pattern,
                               save_dir,
                               gen_sample_limit=gen_sample_limit, 
                               train_sample_limit = train_sample_limit,
                               case_sensitive=case_sensitive)
    elif cmpt=='n':
#         gen_include_key = "SUNet" # input
#         save_dir = f'eval_LS_box/{gen_include_key}_phi_bias'
#         save_file = f'eval_LS_box/{gen_include_key}_phi_bias/stats_params_{gen_include_key}.yaml'
#         print(f'save_dir:{save_dir}')
#         print(f'save_file:{save_file}')

#         gen_include_key = "SPUNet"
#         save_dir = f"eval_LS_box/{gen_include_key}_phi" 
#         save_file = f'eval_LS_box/{gen_include_key}_phi/stats_params_{gen_include_key}.yaml'
#         print(f'save_dir:{save_dir}')
#         print(f'save_file:{save_file}')

        gen_include_key = "SPUNet"
        save_dir = f"eval_LS_box/{gen_include_key}_phi_bias" 
        save_file = f'eval_LS_box/{gen_include_key}_phi_bias/stats_params_{gen_include_key}.yaml'
        print(f'save_dir:{save_dir}')
        print(f'save_file:{save_file}')
        
    # ------------------- 加载数据 ------------------- #
    data = load_yaml_data(save_file)

    # ------------------- 绘制孔隙度箱式图 ------------------- #
    try:
        # porosity_plot_path = os.path.join(save_dir, f"porosity_boxplot_{gen_include_key}_modify.png")
        train_stats, gen_stats = extract_porosity_data(data)
        plot_porosity_boxplot(
            train_stats=train_stats,
            gen_stats=gen_stats,
            save_path=f"{save_dir}/porosity_boxplot.png"
        )
    except ValueError as e:
        print(f"❌ 孔隙度箱式图绘制失败：{e}")

    # ------------------- 绘制原始TPF对比图 ------------------- #
    try:
        tpf_data = extract_tpf_data(data, norm=False)
        plot_tpf_comparison(
            tpf_data=tpf_data,
            save_path=f"{save_dir}/tpf_comparison.png",
            norm=False,
        )
    except ValueError as e:
        print(f"❌ 原始TPF对比图绘制失败：{e}")

    # ------------------- 绘制归一化TPF对比图 ------------------- #
    try:
        norm_tpf_data = extract_tpf_data(data, norm=True)
        plot_tpf_comparison(
            tpf_data=norm_tpf_data,
            save_path=f"{save_dir}/norm_tpf_comparison.png",
            norm=True,
        )
    except ValueError as e:
        print(f"❌ 归一化TPF对比图绘制失败：{e}")

    # ------------------- 绘制LPF对比图 ------------------- #
    try:
        lpf_data = extract_lpf_data(data)
        plot_lpf_comparison(
            lpf_data=lpf_data,
            save_path=f"{save_dir}/lpf_comparison.png"
        )
    except ValueError as e:
        print(f"❌ LPF对比图绘制失败：{e}")

    print("\n# ------------------- 所有绘图流程完成 ------------------- #")
    
    # ------------------- 多数据共一张图 ------------------- #
    datas = []
    save_files = []
    for save_file in save_files:
        data = load_yaml_data(save_file)
        datas.append(data)
        

        
        
        
        