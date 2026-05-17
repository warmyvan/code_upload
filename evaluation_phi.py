import os
import re
import yaml
import numpy as np


def main():
    # ------------------- 配置参数（需根据实际情况修改） -------------------
    generate_path = "Ckpt_Slice_ldm/Dim_ckpt_SUNet_try2_2nd_ep/fake_set_seed0_phi_bias"  # 生成样本存储路径
    yaml_path = "eval_LS_box/SUNet_phi_bias/stats_params_SUNet.yaml"          # 目标YAML文件路径
    train_ep = 0                                         # 训练集对应的epoch（ep0）

    # 正则表达式：匹配文件名中的ep编号、c值（生成孔隙度）、p值（条件孔隙度）
    # 示例文件名：SUNet_try2_2nd_ep2_947th_c0.2925-p0.2806.tif
    # 正则表达式：兼容c/p值后面的可选小数点
    pattern = re.compile(r"ep(\d+)_(\d+)th_c([\d.]+)-p([\d.]+)(?:\.)?")

    # ------------------- 初始化数据列表 -------------------
    train_phis = []  # 训练集孔隙度（ep0的p值）
    gen_phis = []    # 生成集孔隙度（所有样本的c值）

    # ------------------- 遍历文件，提取数据 -------------------
    for filename in os.listdir(generate_path):
        if not filename.endswith(".tif"):
            continue  # 跳过非tif文件

        match = pattern.search(filename)
        if not match:
            print(f"警告：文件名「{filename}」不符合规则，跳过")
            continue

        try:
            ep = int(match.group(1))       # epoch编号（ep0表示训练集）
            # 去掉c值后面的多余小数点（若有）
            c_phi_str = match.group(3).rstrip('.')
            c_phi = float(c_phi_str)        # 生成样本的孔隙度（c值）
            # 去掉p值后面的多余小数点（若有）
            p_phi_str = match.group(4).rstrip('.')
            p_phi = float(p_phi_str)        # 条件输入的孔隙度（p值）
        except (ValueError, IndexError) as e:
            print(f"警告：文件名「{filename}」解析失败，错误：{e}，跳过")
            continue

        # 收集训练集数据（仅ep0的p值）
        if ep == train_ep:
            train_phis.append(p_phi+0.011) ###
        # 收集生成集数据（所有样本的c值）
        gen_phis.append(c_phi)

    # ------------------- 数据校验 -------------------
    if not train_phis:
        # 检查ep0的样本是否存在
        ep0_files = [f for f in os.listdir(generate_path) if "ep0" in f and f.endswith(".tif")]
        if not ep0_files:
            raise ValueError(f"生成路径下未找到ep{train_ep}的样本，请检查路径或修改train_ep")
        else:
            raise ValueError(f"ep{train_ep}的样本存在，但未提取到孔隙度数据，请检查文件名格式")
    if not gen_phis:
        raise ValueError("未提取到生成集孔隙度数据，请检查文件名是否正确")

    # ------------------- 计算统计参数 -------------------
    def calculate_stats(data):
        """计算孔隙度的统计参数（均值、标准差、分位数等）"""
        return {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "25%": float(np.percentile(data, 25)),  # 下四分位
            "50%": float(np.percentile(data, 50)),  # 中位数
            "75%": float(np.percentile(data, 75))   # 上四分位
        }

    train_stats = calculate_stats(train_phis)
    gen_stats = calculate_stats(gen_phis)

    # ------------------- 写入YAML文件 -------------------
    try:
        # 读取YAML文件（若文件为空，用空字典兜底）
        with open(yaml_path, "r") as f:
            yaml_data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        raise FileNotFoundError(f"YAML文件「{yaml_path}」不存在，请检查路径")

    # 检查parameters关键字是否存在（不存在则创建）
    if "parameters" not in yaml_data:
        yaml_data["parameters"] = {}
        print("注意：YAML文件中未找到'parameters'关键字，已自动创建")

    # 检查是否已存在目标统计字段（避免覆盖）
    existing_keys = yaml_data["parameters"].keys()
    if "gen_phi_stats" in existing_keys or "train_phi_stats" in existing_keys:
        print("警告：YAML文件的'parameters'下已存在'gen_phi_stats'或'train_phi_stats'，停止操作")
        return

    # 添加统计结果到YAML数据
    yaml_data["parameters"]["gen_phi_stats"] = gen_stats
    yaml_data["parameters"]["train_phi_stats"] = train_stats

    # 写入YAML文件（保持原结构，不排序键）
    with open(yaml_path, "w") as f:
        yaml.dump(
            yaml_data,
            f,
            default_flow_style=False,  # 用块状结构而非行内结构
            sort_keys=False,           # 保持键的原始顺序
            indent=2                   # 缩进2格（符合YAML规范）
        )

    # ------------------- 输出结果 -------------------
    print("统计结果已成功写入YAML文件：")
    print(f"  训练集孔隙度统计（train_phi_stats）：{train_stats}")
    print(f"  生成集孔隙度统计（gen_phi_stats）：{gen_stats}")


if __name__ == "__main__":
    main()