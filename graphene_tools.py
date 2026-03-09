# graphene_tools.py (适配 Physics-Augmented Log Learning)
import json
import io
import base64
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from langchain.tools import tool

# 引入特征计算
from graphene_features import enhance_features, calculate_theoretical_k

# === 全局配置 ===
MODEL_PATH = "advanced_model.pkl" 
SCALER_PATH = "feature_scaler.pkl"
FEATURE_PATH = "model_features.json"

_gpr_model = None
_scaler = None
_model_features = None

def load_resources():
    """加载资源 (单例模式)"""
    global _gpr_model, _scaler, _model_features
    if _model_features is None:
        try:
            with open(FEATURE_PATH, "r", encoding='utf-8') as f:
                _model_features = json.load(f)
            _scaler = joblib.load(SCALER_PATH)
            _gpr_model = joblib.load(MODEL_PATH)
        except Exception as e:
            return None, None, None, f"资源加载失败: {str(e)}"
    return _gpr_model, _scaler, _model_features, ""

def _predict_core(length_um, temperature_k, defect_ratio, layers=1, doping=0.0, substrate='Suspended'):
    """核心预测引擎：包含特征对齐与物理还原"""
    global _gpr_model, _scaler, _model_features
    
    # ================= 🔥 核心修复点 =================
    # 如果检测到内存里还没有模型，自动加载，防止任何外层应用忘记调用！
    if _gpr_model is None or _scaler is None or _model_features is None:
        _, _, _, err = load_resources()
        if err:
            return 0, 0, 0, f"模型加载致命错误: {err}"
    # ===============================================

    # 🚨 1. 物理适用域护栏检查
    warning_msg = ""
    if layers > 10 or defect_ratio > 0.1:
        warning_msg = "\n⚠️ **[物理边界警告]** 输入参数超出了当前 GPR 模型的适用域 (层数需 ≤10, 缺陷率需 ≤10%)。热传导机制可能已发生本质改变，当前预测值仅供定性参考！"

    # 2. 构建输入数据 (补全所有必需的基础特征)
    # ... 后面的代码保持完全不变 ...
    df_input = pd.DataFrame([{
        'temperature': temperature_k,
        'length_um': length_um,
        'length_nm': length_um * 1000,
        'layers': layers,
        'c12_purity': 0.9999,  # 默认高纯度
        'defect_ratio': defect_ratio,
        'strain': 0.0,
        'doping_concentration': doping,
        'is_suspended': True if substrate == 'Suspended' else False,
        'substrate_type': substrate,
        'defect_topology': 'Pristine' if defect_ratio == 0 else 'Vacancy',
        'doping_type': 'None' if doping == 0 else 'Unknown'
    }])

    # 3. 特征工程与理论基准计算
    df_enhanced = enhance_features(df_input)
    
    raw_theory = calculate_theoretical_k(df_enhanced)
    theory_k = raw_theory.values[0] if hasattr(raw_theory, 'values') else raw_theory

    # 4. 🚀 强制特征对齐 (reindex 装甲，防止模型报错)
    X = pd.get_dummies(df_enhanced, drop_first=False)
    X_reindex = X.reindex(columns=_model_features, fill_value=0)

    # 5. 标准化与 GPR 预测 (获取预测值和不确定度)
    X_scaled = _scaler.transform(X_reindex)
    y_pred_log, y_std_log = _gpr_model.predict(X_scaled, return_std=True)

    # 6. 还原物理真实值 ( k = 理论值 * 修正系数 )
    k_pred = 10**y_pred_log[0] * (theory_k + 1.0)

    return k_pred, y_std_log[0], theory_k, warning_msg

@tool
def ml_prediction_tool(temperature_k: float, length_um: float, defect_ratio: float, layers: int = 1, substrate: str = 'Suspended', **kwargs) -> str:
    """[核心预测] 当用户提供材料参数时，调用此工具预测真实热导率。"""
    try:
        k_pred, std, theory_k, warning = _predict_core(length_um, temperature_k, defect_ratio, layers, 0.0, substrate)
        
        # 将 GPR 的 log 不确定度粗略映射为数值波动范围示意
        uncertainty = k_pred * std * 0.15 
        
        result = (
            f"📊 **预测结果**:\n"
            f"- 预测真实热导率: **{k_pred:.2f} ± {uncertainty:.2f} W/mK**\n"
            f"- 纯理论基准值: {theory_k:.2f} W/mK\n"
            f"- 模型修正幅度: {(k_pred - theory_k):.2f} W/mK\n"
            f"{warning}"
        )
        return result
    except Exception as e:
        return f"预测失败，内部错误: {str(e)}"

@tool
def inverse_design_tool(target_k: float, length_um: float, temperature_k: float) -> str:
    """[逆向设计] 已知目标热导率，反推需要的‘缺陷浓度’上限。"""
    try:
        # 搜索范围
        def objective(defect):
            if defect < 0 or defect > 0.05: return 1e6
            # 🚨 修复：加一个 _ 接收 warning_msg
            pred, _, _, _ = _predict_core(length_um, temperature_k, defect)
            return abs(pred - target_k)

        res = minimize_scalar(objective, bounds=(0.0, 0.05), method='bounded')
        
        if res.success:
            found_defect = res.x
            # 🚨 修复：加一个 _ 接收 warning_msg
            final_k, _, _, _ = _predict_core(length_um, temperature_k, found_defect)
            
            if abs(final_k - target_k) > target_k * 0.2:
                return f"难以达到 {target_k} W/mK。即使缺陷为0，预测值也仅为 {final_k:.1f} W/mK。"
            
            return (f"为了达到 {target_k} W/mK，建议控制缺陷浓度在 {found_defect*100:.4f}% 左右。\n"
                    f"(预测值: {final_k:.1f} W/mK)")
        else:
            return "计算未收敛，目标值可能超出物理极限。"
            
    except Exception as e:
        return f"逆向设计出错: {e}"

@tool
def plot_trend_tool(variable: str, fixed_params: str) -> str:
    """[可视化] 绘制热导率随变量变化的趋势图。"""
    try:
        params = json.loads(fixed_params)
        length = params.get('length_um', 10.0)
        temp = params.get('temperature', 300.0)
        defect = params.get('defect_ratio', 0.001)
        
        x_vals = []
        y_vals = []
        theory_vals = []
        x_label = ""
        
        if variable == 'temperature':
            x_vals = np.linspace(100, 600, 20)
            x_label = "Temperature (K)"
            for t in x_vals:
                # 🚨 修复：加一个 _ 接收 warning_msg
                k, _, th, _ = _predict_core(length, t, defect)
                y_vals.append(k)
                theory_vals.append(th)
        elif variable == 'defect':
            x_vals = np.linspace(0.0, 0.02, 20)
            x_label = "Defect Ratio"
            for d in x_vals:
                # 🚨 修复：加一个 _ 接收 warning_msg
                k, _, th, _ = _predict_core(length, temp, d)
                y_vals.append(k)
                theory_vals.append(th)
        elif variable == 'length':
            x_vals = np.linspace(1.0, 50.0, 20)
            x_label = "Length (um)"
            for l in x_vals:
                # 🚨 修复：加一个 _ 接收 warning_msg
                k, _, th, _ = _predict_core(l, temp, defect)
                y_vals.append(k)
                theory_vals.append(th)
        else:
            return "不支持的变量类型"

        # 🚨 我们之前提到的线程安全面向对象画图法也融合进来了
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x_vals, y_vals, 'o-', color='#d62728', linewidth=2, label='AI Prediction')
        ax.plot(x_vals, theory_vals, '--', color='gray', alpha=0.6, label='Physics Formula')
        
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Thermal Conductivity (W/mK)")
        ax.set_title(f"Trend Analysis ({variable})")
        ax.legend()
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig) # 显式释放内存

        return f"![trend_plot](data:image/png;base64,{img_str})"

    except Exception as e:
        return f"绘图失败: {e}"

@tool
def physics_calculation_tool(temperature_k: float, defect_ratio: float, length_um: float = 10.0, **kwargs) -> str:
    """[物理公式] 仅计算纯物理理论值。"""
    try:
        temp_df = pd.DataFrame([{
            'temperature': temperature_k,
            'defect_ratio': defect_ratio,
            'length_um': length_um,
            'substrate_type': 'Suspended' 
        }])
        k_val, components = calculate_theoretical_k(temp_df, return_components=True)
        analysis_data = {
            "理论上限 (W/mK)": round(k_val[0], 2),
            "机制拆解": {
                "声子散射": round(components['temp_factor'], 3),
                "边界散射": round(components['size_factor'], 3),
                "缺陷散射": round(components['defect_factor'], 3)
            }
        }
        return f"计算成功: {json.dumps(analysis_data, ensure_ascii=False)}"
    except Exception as e:

        return f"物理计算出错: {str(e)}"
