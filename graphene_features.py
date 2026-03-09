import numpy as np
import pandas as pd

def get_substrate_limit(substrate_type):
    """
    根据基底类型返回物理极限 (W/mK)。
    数据来源参考: Seol et al. Science (2010) regarding SPP scattering.
    """
    # 归一化输入字符串
    sub = str(substrate_type).strip()
    
    mapping = {
        # 悬空石墨烯：主要受限于样品尺寸和本征Umklapp散射
        # 理论上限极高，设为 8000-10000 允许模型自由探索
        'Suspended': 10000.0,
        
        # 氮化硼 (hBN)：原子平整，声子模式匹配较好，干扰较小
        'hBN': 2000.0,
        
        # 二氧化硅 (SiO2)：强烈的表面极化子(SPP)散射，这是硬物理上限
        'SiO2': 600.0,
        
        # 金属基底：强烈的电子-声子耦合导致声子寿命骤减
        'Au': 200.0,
        'Cu': 200.0,
        'Ni': 100.0
    }
    # 默认值给一个比较保守的基底值 (如 SiO2 类似)
    return mapping.get(sub, 500.0)

def calculate_theoretical_k(df, return_components=False):
    """
    计算理论热导率 (Alex 修正版：动态基底约束)
    采用 Matthiessen's Rule: 1/k_total = 1/k_intrinsic + 1/k_substrate
    """
    # 1. 提取物理量，处理 Series 或单个值的情况
    T = df.get('temperature', 300.0)
    L = df.get('length_um', 10.0)
    defect = df.get('defect_ratio', 0.0)
    
    # 提取基底类型 (如果是 DataFrame 则由 map 处理，如果是单行字典则直接 get)
    if isinstance(df, pd.DataFrame):
        # 向量化操作
        if 'substrate_type' in df.columns:
            k_sub_limit = df['substrate_type'].apply(get_substrate_limit).values
        else:
            k_sub_limit = np.full(len(df), 500.0) # 默认最差情况
    else:
        # 单次推理 (Series 或 Dict)
        sub_type = df.get('substrate_type', 'SiO2')
        k_sub_limit = get_substrate_limit(sub_type)

    # === A. 本征部分 (Intrinsic) ===
    # 修正 Klemens-Callaway 近似
    
    # 缺陷因子 (Stone-Wales 缺陷散射)
    # 缺陷会引入巨大的热阻，使用指数衰减模拟
    defect_factor = 1.0 / (1.0 + 5000.0 * defect) 
    
    # 温度因子 (Umklapp 散射: ~ 1/T)
    # 在低温区 (<150K) 受边界限制，高温区 (>300K) 受 U 散射限制
    # 这里做一个简单的平滑过渡
    temp_factor = (300.0 / (T + 10.0)) ** 1.2
    
    # 尺寸因子 (Ballistic to Diffusive crossover)
    # log 关系在宏观尺度合适，但在纳米尺度最好用 L/(L+MFP)
    # 假设平均自由程 MFP_0 ~ 0.7 um @ 300K
    mfp_approx = 0.7 * (300.0 / T)
    size_factor = L / (L + mfp_approx) * 5.0 # 归一化因子

    # 理想本征热导率 (Suspended & Defect-free)
    # 基准常数设为 4000 (室温悬空石墨烯的典型值)
    base_k_intrinsic = 4000.0 * temp_factor * size_factor * defect_factor

    # === B. 外部限制 (Extrinsic) ===
    # 使用马西森定则合并：总热阻 = 本征热阻 + 基底热阻
    # k_total = (k_int * k_sub) / (k_int + k_sub)
    
    final_k = (base_k_intrinsic * k_sub_limit) / (base_k_intrinsic + k_sub_limit)
    
    # 防止数值下溢
    final_k = np.maximum(final_k, 5.0)

    if return_components:
        # 调试用：查看各部分贡献
        # 注意：如果是向量化调用，这里只返回均值供参考
        return final_k, {
            "intrinsic_k_mean": np.mean(base_k_intrinsic),
            "substrate_limit_mean": np.mean(k_sub_limit),
            "defect_impact": np.mean(defect_factor)
        }
    
    return final_k

def enhance_features(df):
    """特征工程管道 (保持你的原逻辑，加上一点微调)"""
    df_out = df.copy()
    
    if 'temperature' in df_out.columns:
        df_out['log_temp'] = np.log10(df_out['temperature'] + 1.0)
    if 'length_um' in df_out.columns:
        df_out['log_length'] = np.log10(df_out['length_um'] + 0.001)
    if 'defect_ratio' in df_out.columns:
        df_out['log_defect'] = np.log10(df_out['defect_ratio'] + 1e-9)

    # 简单的物理因子
    df_out['iso_factor'] = 1.0
    df_out['chem_factor'] = 1.0

    # 计算修正后的理论值
    raw_theory_k = calculate_theoretical_k(df_out, return_components=False)
    
    # 处理基底 (Substrate)
    if 'substrate_type' in df_out.columns:
        sub_map = {'Suspended': 1.0, 'hBN': 0.8, 'SiO2': 0.4, 'Au': 0.1, 'Cu': 0.1}
        substrate_factor = df_out['substrate_type'].map(sub_map).fillna(0.4)
    else:
        substrate_factor = 0.4 # 默认认为有基底干扰

    combined_factor = substrate_factor
    
    # 特征里也存一份 log_theory
    df_out['log_theory_k'] = np.log10(raw_theory_k * combined_factor + 1.0)
    
    return df_out