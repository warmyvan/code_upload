from LS_Box_multi import *

if __name__ == "__main__":
    # 配置参数
    model_configs = [
        {"name": "Model_A", "yaml_path": "results/model_a_stats.yaml"},
        {"name": "Model_B", "yaml_path": "results/model_b_stats.yaml"},
        {"name": "Model_C", "yaml_path": "results/model_c_stats.yaml"}
    ]
    
    # 1. 加载所有模型数据
    yaml_paths = [cfg["yaml_path"] for cfg in model_configs]
    model_labels = [cfg["name"] for cfg in model_configs]
    
    train_data, gen_data_list = load_multiple_yaml_data(yaml_paths)
    
    # 2. 提取训练集数据
    train_porosity_stats = extract_porosity_data(train_data)[0]
    train_tpf = extract_tpf_data(train_data, norm=False)['train']
    train_norm_tpf = extract_tpf_data(train_data, norm=True)['train']
    train_lpf = extract_lpf_data(train_data)['train']
    
    # 3. 提取生成集孔隙度统计（列表）
    gen_porosity_stats_list = [
        extract_porosity_data(gen_data)[1] 
        for gen_data in gen_data_list
    ]
    
    # 4. 绘制多模型对比图
    save_dir = "multi_model_comparison"
    os.makedirs(save_dir, exist_ok=True)
    
    # (1) 多模型孔隙度箱式图
    plot_multi_porosity_boxplot(
        train_stats=train_porosity_stats,
        gen_stats_list=gen_porosity_stats_list,
        model_labels=model_labels,
        save_path=f"{save_dir}/multi_porosity_boxplot.png"
    )
    
    # (2) 多模型原始TPF对比图
    plot_multi_correlation_comparison(
        train_data=train_tpf,
        gen_data_list=gen_data_list,
        model_labels=model_labels,
        save_path=f"{save_dir}/multi_tpf_comparison.png",
        correlation_type='tpf',
        x_label='Distance',
        y_label='$S_2(r)$',
        title='Multi-Model Two-Point Correlation Function Comparison'
    )
    
    # (3) 多模型归一化TPF对比图
    plot_multi_correlation_comparison(
        train_data=train_norm_tpf,
        gen_data_list=gen_data_list,
        model_labels=model_labels,
        save_path=f"{save_dir}/multi_norm_tpf_comparison.png",
        correlation_type='norm_tpf',
        x_label='Distance',
        y_label='$S_2(r)/\phi$',
        title='Multi-Model Normalized Two-Point Correlation Function Comparison'
    )
    
    # (4) 多模型LPF对比图
    plot_multi_correlation_comparison(
        train_data=train_lpf,
        gen_data_list=gen_data_list,
        model_labels=model_labels,
        save_path=f"{save_dir}/multi_lpf_comparison.png",
        correlation_type='lpf',
        x_label='Path Length',
        y_label='LPF Probability',
        title='Multi-Model Linear Path Function Comparison'
    )
    
    print("\n# ------------------- 所有多模型对比图已完成 ------------------- #")