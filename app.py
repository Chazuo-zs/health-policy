"""
健康决策支持系统 - Streamlit 前端
================================
基于 XGBoost 健康预测模型 + 多期优化算法
提供国家画像、模型优化两大功能，支持大模型辅助决策
"""

import os
import sys
import warnings
import functools
import threading
import time
import json
import re

# 加载 .env 文件（如果存在）
from pathlib import Path
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    with open(_env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()
from io import StringIO

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
from scipy.optimize import minimize

# ─────────────────────────────────────────────
# 大模型 API 配置（用户可自行填写）
# ─────────────────────────────────────────────
LLM_CONFIG = {
    "enabled": True,                        # 设置为 True 启用大模型辅助
    "provider": "deepseek",                 # 提供商: "deepseek", "openai", "zhipu", "kimi"
    "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),  # 从环境变量读取，更安全
    "base_url": "https://api.deepseek.com/v1",  # API 地址（OpenAI 兼容格式）
    "model": "deepseek-chat",               # 模型名称
    "timeout": 60,                          # 请求超时（秒）
}

def call_llm_recommend_constraints(country_name: str, country_data: dict,
                                     feature_labels: dict, cluster_type: str = "Mid-Health") -> dict:
    """
    调用大模型推荐约束系数

    Args:
        country_name: 国家名称
        country_data: 该国家当前指标数据
        feature_labels: 指标中英文映射
        cluster_type: 聚类类型 "Low-Health" / "Mid-Health" / "High-Health"
    Returns:
        dict: 每个指标的推荐约束系数 {"pm25_exposure": {"max_annual": 3.0}, ...}
    """
    if not LLM_CONFIG["enabled"] or not LLM_CONFIG["api_key"]:
        return {}

    try:
        import httpx

        # 判断是否是High-Health国家（已经是高健康水平）
        is_high_health = cluster_type == "High-Health"

        prompt = f"""你是一个健康政策专家。请根据以下国家的信息，推荐该国每个健康指标每年合理的最大改善幅度（约束）。

【国家基本信息】
国家名称: {country_name}
所属层级: {cluster_type}

【当前指标数据】
{json.dumps(country_data, ensure_ascii=False, indent=2)}

【指标详细说明】
| 指标 | 说明 | 含义 |
|------|------|------|
| pm25_exposure | PM2.5年暴露量 (μg/m³) | 越低越好，WHO建议<5μg/m³ |
| safe_water | 安全饮用水覆盖率 (%) | 越高越好 |
| basic_sanitation | 基础卫生设施覆盖率 (%) | 越高越好 |
| gdp_per_capita_ppp | 人均GDP (PPP, 美元) | 反映经济水平 |
| physicians_per_1000 | 每千人医生数 | 反映医疗资源 |
| beds_per_1000 | 每千人床位数 | 反映医疗资源 |
| health_exp_gdp | 卫生支出占GDP比例 (%) | 反映卫生投入 |
| elderly_share | 老年人口比例 (%) | 老龄化程度>20%为深度老龄化 |
| G_norm | 治理指数 (0-1) | 反映治理能力 |

【输出格式要求】
请只输出一个合法的JSON对象（不要输出任何其他内容）：
{{
    "pm25_exposure": {{"max_annual": -1.0, "note": "..."}},
    "safe_water": {{"max_annual": 3.0, "note": "..."}},
    "basic_sanitation": {{"max_annual": 3.0, "note": "..."}},
    "gdp_per_capita_ppp": {{"max_annual": 0, "note": "..."}},
    "physicians_per_1000": {{"max_annual": 0.1, "note": "..."}},
    "beds_per_1000": {{"max_annual": 0.2, "note": "..."}},
    "health_exp_gdp": {{"max_annual": 0.3, "note": "..."}},
    "elderly_share": {{"max_annual": 0, "note": "..."}},
    "G_norm": {{"max_annual": 0.02, "note": "..."}}
}}

【必须遵守的规则】
1. gdp_per_capita_ppp（人均GDP）和 elderly_share（老年人口比例）必须设为0，不建议主动干预
2. pm25_exposure 必须设为负值（表示降低暴露量）
3. 其他正向指标必须设为正值（表示提升）
4. 所有 max_annual 值必须是数值，不能是 null、None 或字符串

【各层级国家的改善潜力指导】

▌Low-Health（低健康水平国家，如印度、尼日利亚、埃塞俄比亚等）
特征：基础设施薄弱，医疗资源匮乏，PM2.5污染严重，但改善空间巨大
建议改善空间：
- pm25_exposure: -8.0 ~ -3.0（工业化初期，污染治理力度大）
- safe_water: 5.0 ~ 15.0（基础设施缺口大，可大幅提升）
- basic_sanitation: 5.0 ~ 15.0（卫生设施严重不足，改善空间大）
- physicians_per_1000: 0.1 ~ 0.5（医疗人才严重短缺，可大幅补充）
- beds_per_1000: 0.2 ~ 1.0（床位资源严重不足）
- health_exp_gdp: 0.5 ~ 2.0（卫生投入增长空间大）
- G_norm: 0.03 ~ 0.08（治理能力提升空间大）
典型案例：印度从2000年PM2.5=60降到2019年≈13，每年约需下降2-3

▌Mid-Health（中等健康水平国家，如中国、巴西、泰国等）
特征：基础设施初具规模，医疗资源中等，面临环境污染和医疗资源不均问题
建议改善空间：
- pm25_exposure: -5.0 ~ -1.0（大气污染防治攻坚期）
- safe_water: 2.0 ~ 5.0（从"有"到"好"的提升）
- basic_sanitation: 2.0 ~ 5.0（质量提升而非数量扩张）
- physicians_per_1000: 0.05 ~ 0.2（人才质量提升为主）
- beds_per_1000: 0.1 ~ 0.5（优化结构而非简单扩张）
- health_exp_gdp: 0.2 ~ 0.8（效率提升为主）
- G_norm: 0.01 ~ 0.05（治理精细化）
典型案例：中国近年来医改年均医生增长约0.05-0.1/1000

▌High-Health（高健康水平国家，如日本、瑞士、新加坡、德国等）
特征：基础医疗设施完善，老龄化严重，面临医疗效率和质量提升挑战
建议改善空间（即使已达全球领先水平，也必须给出有意义的空间）：
- pm25_exposure: -0.5 ~ -2.0（追求WHO标准<5μg/m³）
- safe_water: 0.1 ~ 0.5（惠及少数未覆盖人群）
- basic_sanitation: 0.1 ~ 0.5（质量持续改进）
- physicians_per_1000: 0.03 ~ 0.15（应对老龄化，保持医疗可及性）
- beds_per_1000: 0.05 ~ 0.3（优化床位周转效率）
- health_exp_gdp: 0.1 ~ 0.5（老龄化应对成本）
- G_norm: 0.005 ~ 0.03（公共服务精细化）
重要提醒：即使日本PM2.5已达12μg/m³、医生达2.5/1000，也要给出改善空间！
原因：老龄化加剧→需更多医疗资源；治理标准提升→持续改进

【综合判断原则】
1. 根据该国当前指标实际值判断：
   - 若某指标已接近理论上限（如safe_water>98%），改善空间相应减小但不为0
   - 若某指标仍有较大差距，改善空间可适度放大
2. 考虑该国的历史发展趋势和经济约束
3. 确保约束值能让优化器在10年内实现有意义的HALE提升
4. 避免给出过于保守的值（如接近0），这会使优化完全失效"""

        headers = {
            "Authorization": f"Bearer {LLM_CONFIG['api_key']}",
            "Content-Type": "application/json",
        }

        messages = [
            {"role": "system", "content": "你是一个专业的健康政策顾问。请严格按要求输出JSON，不要有任何额外文字。"},
            {"role": "user", "content": prompt}
        ]

        payload = {
            "model": LLM_CONFIG["model"],
            "messages": messages,
            "temperature": 0.3,
        }

        base_url = LLM_CONFIG.get("base_url", "https://api.openai.com/v1")
        with httpx.Client(timeout=LLM_CONFIG.get("timeout", 60)) as client:
            response = client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]

        # 提取JSON
        content = re.search(r'\{[\s\S]*\}', content)
        if content:
            result_dict = json.loads(content.group(0))
            # 确保所有值都是数值
            for feat in result_dict:
                if isinstance(result_dict[feat], dict) and "max_annual" in result_dict[feat]:
                    try:
                        result_dict[feat]["max_annual"] = float(result_dict[feat]["max_annual"])
                    except (ValueError, TypeError):
                        result_dict[feat]["max_annual"] = 0.0
            return result_dict
    except Exception as e:
        st.warning(f"大模型调用失败: {e}")
        return {}

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120

# ─────────────────────────────────────────────
# 自定义现代主题样式
# ─────────────────────────────────────────────
MODERN_CSS = """
<style>
/* ===== 全局 ===== */
.stApp {
    font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
    background: #f0fdf4;
}

/* ===== 主标题 ===== */
.main-title {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #059669 0%, #10b981 50%, #0284c7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem !important;
}

/* ===== 副标题 ===== */
.sub-title {
    color: #059669;
    font-size: 1.1rem;
    margin-bottom: 1.5rem !important;
}

/* ===== 指标卡片 ===== */
.metric-card {
    background: white;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 4px 20px rgba(5, 150, 105, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.2);
    transition: all 0.3s ease;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 30px rgba(16, 185, 129, 0.2);
}

/* ===== 聚类标签 ===== */
.cluster-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.9rem;
    background: linear-gradient(135deg, #059669 0%, #10b981 100%);
    color: white;
}

/* ===== 图表容器 ===== */
.chart-container {
    background: white;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 4px 20px rgba(5, 150, 105, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.15);
    margin: 10px 0;
}

/* ===== 侧边栏 - 深绿渐变 ===== */
.css-1d391kg {
    background: linear-gradient(180deg, #064e3b 0%, #059669 50%, #047857 100%);
}
.css-1d391kg, .css-1d391kg * { color: white !important; }
.css-qrbaxs, .css-qrbaxs * { color: white !important; }

/* ===== 导航项 ===== */
.nav-item {
    padding: 12px 16px;
    border-radius: 12px;
    margin: 6px 0;
    transition: all 0.3s ease;
    cursor: pointer;
    color: white;
}
.nav-item:hover { background: rgba(16, 185, 129, 0.3); transform: translateX(4px); }

/* ===== 区块标题 ===== */
.section-header {
    font-size: 1.4rem;
    font-weight: 600;
    color: #047857;
    margin: 1.5rem 0 1rem 0;
    padding-left: 16px;
    border-left: 4px solid #10b981;
}

/* ===== 数据表格 ===== */
.data-table { background: white; border-radius: 12px; overflow: hidden; }

/* ===== 输入区域 ===== */
.input-section {
    background: white;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 4px 20px rgba(5, 150, 105, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.15);
    margin: 15px 0;
}

/* ===== 按钮 ===== */
.stButton > button {
    border-radius: 12px;
    padding: 12px 24px;
    font-weight: 600;
    background: linear-gradient(135deg, #059669 0%, #10b981 100%);
    color: white;
    border: none;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #047857 0%, #059669 100%);
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(5, 150, 105, 0.4);
}

/* ===== 提示框 ===== */
.success-box { background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border-radius: 12px; padding: 16px; border-left: 4px solid #059669; color: #047857; }
.warning-box { background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-radius: 12px; padding: 16px; border-left: 4px solid #d97706; color: #92400e; }
.info-box { background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border-radius: 12px; padding: 16px; border-left: 4px solid #0284c7; color: #1e40af; }

/* ===== 进度条 ===== */
.stProgress > div > div { background: linear-gradient(90deg, #059669 0%, #10b981 50%, #0284c7 100%); border-radius: 10px; }

/* ===== 侧边栏标题 ===== */
.sidebar-header { text-align: center; padding: 20px 0; border-bottom: 1px solid rgba(255,255,255,0.2); margin-bottom: 20px; }
.sidebar-header h2 { color: white; font-size: 1.5rem; margin: 0; }

/* ===== 图标 ===== */
.nav-icon { font-size: 1.2rem; margin-right: 10px; }

/* ===== 分隔线 ===== */
.custom-divider { border: none; height: 1px; background: linear-gradient(90deg, transparent, #10b981, transparent); margin: 20px 0; }

/* ===== 国家信息卡片 ===== */
.country-info-card { background: linear-gradient(135deg, #059669 0%, #10b981 50%, #0284c7 100%); color: white; border-radius: 16px; padding: 20px; margin: 15px 0; }

/* ===== 结果卡片 ===== */
.result-card { background: white; border-radius: 16px; padding: 24px; box-shadow: 0 4px 20px rgba(5, 150, 105, 0.1); text-align: center; transition: all 0.3s ease; border: 1px solid rgba(16, 185, 129, 0.2); }
.result-card:hover { transform: scale(1.02); box-shadow: 0 8px 30px rgba(5, 150, 105, 0.15); }

/* ===== 政策建议框 ===== */
.policy-box { background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); border-radius: 16px; padding: 20px; border-left: 4px solid #10b981; margin: 15px 0; color: #047857; }

/* ===== 展开器 ===== */
.streamlit-expanderHeader { background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); border-radius: 8px; color: #047857; font-weight: 600; }
.streamlit-expanderContent { background: white; border-radius: 0 0 8px 8px; }

/* ===== 隐藏元素 ===== */
.css-1v0mbdj { display: none; }

/* ===== 图表标题 ===== */
.figure-title { font-size: 1.1rem; font-weight: 600; color: #047857; margin-bottom: 15px; text-align: center; }

/* ===== 加载动画 ===== */
.loading-spinner { border: 4px solid rgba(16, 185, 129, 0.1); border-top: 4px solid #059669; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; }
@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

/* ===== 表单输入 ===== */
.stTextInput > div > div > input, .stNumberInput > div > div > input, .stSelectbox > div > div > select { border: 2px solid #10b981; border-radius: 8px; background-color: white; color: #1f2937; }
.stTextInput > div > div > input:focus, .stNumberInput > div > div > input:focus, .stSelectbox > div > div > select:focus { border-color: #059669; box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.2); }

/* ===== 选项卡 ===== */
.stTabs [data-baseweb="tab-list"] { background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); border-radius: 12px; padding: 4px; gap: 8px; }
.stTabs [data-baseweb="tab"] { color: #059669; font-weight: 600; background: white; border-radius: 10px; padding: 10px 20px; }
.stTabs [data-baseweb="tab"]:hover { background: rgba(16, 185, 129, 0.1); }
.stTabs [aria-selected="true"] { background: #059669 !important; color: white !important; }

/* ===== 指标数字 ===== */
[data-testid="stMetricValue"] { color: #059669; font-weight: 700; }
[data-testid="stMetricLabel"] { color: #047857; }

/* ===== 滑块 ===== */
.stSlider > div > div > div { background: #d1fae5; }
.stSlider > div > div > div > div { background: #059669; }

/* ===== 主内容区 ===== */
.main .block-container { background: white; border-radius: 20px; padding: 2rem; box-shadow: 0 10px 40px rgba(5, 150, 105, 0.08); }

/* ===== 全局文字 ===== */
body, .stApp { color: #1f2937; }

/* ===== 分隔线 ===== */
hr { border-color: #10b981; }
[data-testid="stHorizontalBlock"] { border-left: 3px solid #10b981; padding-left: 10px; }

/* ===== 表格 ===== */
.dataframe { border: 1px solid #10b981 !important; border-radius: 8px; }
.dataframe th { background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important; color: white !important; }
.dataframe tr:nth-child(even) { background: #ecfdf5 !important; }
.dataframe tr:hover { background: #d1fae5 !important; }

/* ===== 焦点样式 ===== */
:focus { outline: 2px solid #059669; outline-offset: 2px; }

/* ===== 选中文字 ===== */
::selection { background-color: #10b981; color: white; }

/* ===== 图表 ===== */
matplotlib-figure { border-radius: 12px; overflow: hidden; }

/* ===== 滚动条 ===== */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: #ecfdf5; }
::-webkit-scrollbar-thumb { background: #10b981; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #059669; }
</style>
"""

# ─────────────────────────────────────────────
# 路径配置（相对路径，基于 app.py 所在目录）
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "home", "mw", "out")
SYS_DIR  = os.path.join(BASE_DIR, "home", "mw", "object")



MODEL_PATH   = os.path.join(OUT_DIR, "xgb_model_v2.pkl")
CLUSTER_PATH = os.path.join(OUT_DIR, "country_clusters_v2_k3.csv")
PANEL_PATH   = os.path.join(OUT_DIR, "cleaned_panel.csv")
SUMMARY_PATH = os.path.join(OUT_DIR, "optimization_summary_v3.csv")

FIG_OPT_DIR  = os.path.join(OUT_DIR, "figures_opt_v3")
FIG_PRED_DIR = os.path.join(OUT_DIR, "figures_pred")
FIG_CLS_DIR  = os.path.join(OUT_DIR, "figures_v2")

# ─────────────────────────────────────────────
# 页面配置
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="健康决策支持系统",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

sns.set_style("whitegrid")

# 应用自定义样式
st.markdown(MODERN_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 全局数据加载（session_state 缓存）
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_panel():
    return pd.read_csv(PANEL_PATH)

@st.cache_data
def load_cluster():
    return pd.read_csv(CLUSTER_PATH)

@st.cache_data
def load_summary():
    if os.path.exists(SUMMARY_PATH):
        return pd.read_csv(SUMMARY_PATH)
    return pd.DataFrame()

def save_summary_entry(country, cluster, initial_hale, target_hale,
                        achieved_hale, hale_gain, total_cost, converged):
    row = {
        "Country":       country,
        "Cluster":       cluster,
        "Initial_HALE":  round(initial_hale, 2),
        "Target_HALE":   round(target_hale, 2),
        "Achieved_HALE": round(achieved_hale, 2),
        "HALE_Gain":     round(hale_gain, 2),
        "Total_Cost":    round(total_cost, 6),
        "Converged":     converged,
    }
    df = load_summary()
    existing_mask = df["Country"] == country
    if existing_mask.any():
        df = df[~existing_mask]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(SUMMARY_PATH, index=False)
    try:
        st.cache_data.clear()
        st.session_state.SUMMARY = df
    except Exception:
        pass

# ─────────────────────────────────────────────
# 优化器核心逻辑（从 optimization.py 移植）
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "pm25_exposure", "safe_water", "basic_sanitation",
    "gdp_per_capita_ppp", "physicians_per_1000", "beds_per_1000",
    "health_exp_gdp", "elderly_share", "G_norm"
]
FEATURE_LABELS = {
    "pm25_exposure":       "PM2.5 Exposure",
    "safe_water":          "Safe Water (%)",
    "basic_sanitation":    "Basic Sanitation (%)",
    "gdp_per_capita_ppp":  "GDP per capita (PPP)",
    "physicians_per_1000": "Physicians /1000",
    "beds_per_1000":       "Hospital Beds /1000",
    "health_exp_gdp":      "Health Exp (% GDP)",
    "elderly_share":        "Elderly Share (%)",
    "G_norm":              "Governance Index"
}
N_FEAT = len(FEATURE_COLS)

# ─────────────────────────────────────────────
# 计算每个国家的GDP自然增长率和老年人口比例变化率
# 基于历史数据计算
# ─────────────────────────────────────────────
def compute_country_natural_growth(df_panel_data):
    """
    基于历史数据计算每个国家的GDP年均增长率和老年人口比例年均变化率
    
    GDP增长率计算：
    - 使用复合年增长率(CAGR)公式: g = (V_final/V_initial)^(1/n) - 1
    - 在log空间计算更稳定
    
    老年人口比例变化率：
    - 直接计算年均绝对变化量(百分点/年)
    
    Returns:
        dict: {
            "country_name": {
                "gdp_growth_rate": float,  # GDP年均增长率 (如 0.03 表示3%)
                "elderly_change_rate": float  # 老年人口比例年均变化 (百分点/年)
            }
        }
    """
    growth_rates = {}
    df = df_panel_data.sort_values(["country", "year"])
    
    for country in df["country"].unique():
        country_df = df[df["country"] == country].sort_values("year")
        if len(country_df) < 2:
            growth_rates[country] = {
                "gdp_growth_rate": 0.03,  # 默认3%
                "elderly_change_rate": 0.3  # 默认0.3个百分点
            }
            continue
        
        # 取首尾两年数据计算增长率
        first_row = country_df.iloc[0]
        last_row = country_df.iloc[-1]
        n_years = last_row["year"] - first_row["year"]
        
        if n_years <= 0:
            growth_rates[country] = {
                "gdp_growth_rate": 0.03,
                "elderly_change_rate": 0.3
            }
            continue
        
        # GDP增长率（在log空间计算）
        gdp_initial = first_row["gdp_per_capita_ppp"]
        gdp_final = last_row["gdp_per_capita_ppp"]
        if gdp_initial > 0 and gdp_final > 0:
            # 复合增长率
            gdp_growth = (gdp_final / gdp_initial) ** (1.0 / n_years) - 1.0
            # 限制在合理范围 [-0.1, 0.3]
            gdp_growth = np.clip(gdp_growth, -0.1, 0.3)
        else:
            gdp_growth = 0.03
        
        # 老年人口比例年均变化量（百分点/年）
        elderly_initial = first_row["elderly_share"]
        elderly_final = last_row["elderly_share"]
        elderly_change = (elderly_final - elderly_initial) / n_years
        # 限制在合理范围
        elderly_change = np.clip(elderly_change, -1.0, 2.0)
        
        growth_rates[country] = {
            "gdp_growth_rate": gdp_growth,
            "elderly_change_rate": elderly_change
        }
    
    return growth_rates


def get_country_natural_growth(country_name, country_growth_rates):
    """获取指定国家的自然增长率"""
    if country_name in country_growth_rates:
        return country_growth_rates[country_name]
    return {"gdp_growth_rate": 0.03, "elderly_change_rate": 0.3}


# ─────────────────────────────────────────
# PM2.5 国家级 bounds 修正
# 问题：模型对PM2.5存在倒U型关系，高端国家(如日本)PM2.5已很低，
# 继续降低反而会降低HALE预测，因此PM2.5不能低于该国当前值
# ─────────────────────────────────────────
def make_country_aware_bounds(x0):
    """
    根据各国当前指标值，生成感知国家具体情况的优化边界。

    注意：这个函数用于模拟轨迹时的状态限制。
    对于 PM2.5：
    - 使用全局下界 0（允许降低到 0），而不是 x0 - 0.01
      因为 var_bounds 已经控制了每年的变化幅度，不需要在这里限制
    - 使用全局上界 100
    """
    bnds_low = BOUNDS_LOW.copy()
    bnds_high = BOUNDS_HIGH.copy()
    # PM2.5: 使用全局下界 0（允许降低到 0 μg/m³）
    # var_bounds 已经限制了每期的变化幅度，这里不需要额外限制
    # bnds_low[0] = max(0.0, x0[0] - 0.01)  # 旧逻辑：阻止降低 - 已移除
    # bnds_high[0] = min(100.0, x0[0] + MAX_ANNUAL[0])  # 旧逻辑 - 已移除，使用全局上界
    return bnds_low, bnds_high

BOUNDS_LOW  = np.array([  0,    0,    0,    100,  0,   0,   0,   0,  0.05])
BOUNDS_HIGH = np.array([100,  100,  100, 200000, 10,  20,  20,  30,  1.00])
MAX_ANNUAL  = np.array([5.0, 5.0, 5.0, 3000, 0.15, 0.3, 0.8, 0.5, 0.05])
COST_WEIGHTS = np.array([0.3, 0.8, 0.8, 0.2, 3.0, 2.0, 0.6, 0.05, 1.5])
DISCOUNT_RATE = 0.03
RESTART_TIMEOUT = 300

CLUSTER_PALETTE = {
    "Low-Health":  "#4db89e",
    "Mid-Health":  "#f4a261",
    "High-Health": "#7b9ec7"
}

CLUSTER_GAP = {"Low-Health": 5.0, "Mid-Health": 3.0, "High-Health": 2.0}

# ─────────────────────────────────────────────
# 数据驱动的分组成本权重计算
# 方法：统计各组内各指标的历史年均变化幅度
# 变化越小 → 越难改 → 成本权重越高
# ─────────────────────────────────────────────
def compute_group_cost_weights(panel_df, cluster_df, feature_cols, direction):
    """
    从历史数据计算每个聚类组的成本权重

    Args:
        panel_df: 面板数据 DataFrame
        cluster_df: 国家聚类结果 DataFrame
        feature_cols: 特征列名列表
        direction: 干预方向 (+1=越高越好, -1=越低越好, 0=不干预)

    Returns:
        dict: {"Low-Health": np.array([...]), "Mid-Health": [...], "High-Health": [...]}
    """
    df = panel_df.merge(cluster_df[["country", "gmm_type"]], on="country", how="inner")

    # 只使用有明确聚类标签的数据
    df = df[df["gmm_type"].isin(["Low-Health", "Mid-Health", "High-Health"])]

    # 按国家和年份排序
    df = df.sort_values(["country", "year"])

    # 计算每个国家每个指标的年均变化（跨年度差分）
    change_data = []
    for country in df["country"].unique():
        country_df = df[df["country"] == country].sort_values("year")
        if len(country_df) < 2:
            continue
        for i in range(1, len(country_df)):
            row1 = country_df.iloc[i - 1]
            row2 = country_df.iloc[i]
            years_gap = row2["year"] - row1["year"]
            if years_gap <= 0:
                continue
            gmm_type = row1["gmm_type"]
            for feat in feature_cols:
                # 处理log变换列
                if feat in ["gdp_per_capita_ppp", "physicians_per_1000"]:
                    val1 = np.log(max(row1[feat], 1))
                    val2 = np.log(max(row2[feat], 1))
                else:
                    val1 = row1[feat]
                    val2 = row2[feat]
                annual_change = (val2 - val1) / years_gap
                # 取绝对值，因为成本只看变化幅度
                change_data.append({
                    "country": country,
                    "gmm_type": gmm_type,
                    "feature": feat,
                    "annual_change": abs(annual_change)
                })

    change_df = pd.DataFrame(change_data)

    # 计算每个聚类组每个指标的平均变化幅度
    group_weights = {}
    for gmm_type in ["Low-Health", "Mid-Health", "High-Health"]:
        type_df = change_df[change_df["gmm_type"] == gmm_type]
        weights = np.zeros(len(feature_cols))

        for i, feat in enumerate(feature_cols):
            feat_changes = type_df[type_df["feature"] == feat]["annual_change"].values

            if len(feat_changes) == 0:
                # 没有数据用默认值
                weights[i] = 1.0
            else:
                # 均值变化越小，权重越高
                mean_change = np.mean(feat_changes)
                # 使用 log 变换避免极端值影响
                mean_change = np.log1p(mean_change)
                # 归一化到合理范围（0.1 ~ 5.0）
                weights[i] = 1.0 / (mean_change + 0.01)

        # 归一化：使所有权重和为9（每个特征平均权重为1）
        weights = weights / weights.mean()

        # 根据干预方向调整权重
        # 不干预的指标（direction=0）权重设为0，因为这些是自然变化的指标
        # GDP和老年人口比例是自然变化/背景趋势，不计入政策干预成本
        for i, feat in enumerate(feature_cols):
            feat_idx = FEATURE_COLS.index(feat)
            if direction[feat_idx] == 0:
                weights[i] = 0.0  # 不干预的特征不计成本

        group_weights[gmm_type] = weights

    # 打印结果供验证
    print("\n=== 数据驱动的分组成本权重 ===")
    for gtype, w in group_weights.items():
        print(f"\n{gtype}:")
        for i, feat in enumerate(feature_cols):
            print(f"  {FEATURE_LABELS[feat]:30s}: {w[i]:.4f}")

    return group_weights


def get_cost_weights_for_country(cluster_df, group_cost_weights, country_name):
    """获取指定国家所属组的成本权重"""
    row = cluster_df[cluster_df["country"] == country_name]
    if len(row) == 0:
        return group_cost_weights.get("Mid-Health", COST_WEIGHTS)
    gtype = row.iloc[0]["gmm_type"]
    return group_cost_weights.get(gtype, COST_WEIGHTS)


def apply_log_transform(x_raw):
    x = x_raw.copy()
    x[3] = np.log(max(x[3], 1))
    x[4] = np.log1p(max(x[4], 0))
    return x


@functools.lru_cache(maxsize=500)
def predict_hale_cached(x_raw_tuple, model_id):
    x_raw = np.array(x_raw_tuple)
    x_tf  = apply_log_transform(x_raw)
    return float(MODEL.predict(x_tf.reshape(1, -1))[0])


def get_initial_state(df_panel_data, country_name, year=2019):
    sub = df_panel_data[df_panel_data["country"] == country_name].sort_values("year")
    if sub.empty:
        raise ValueError(f"未找到国家: {country_name}")
    row = sub[sub["year"] <= year].iloc[-1]
    x0  = row[FEATURE_COLS].values.astype(float)
    return x0, float(row["HALE"]), int(row["year"])


def simulate_trajectory(x0, delta_matrix, T, df_panel_data, bounds_low=None, bounds_high=None,
                       natural_growth=None):
    """
    轨迹模拟，默认使用全局bounds，但支持传入国家感知bounds
    
    Args:
        x0: 初始状态
        delta_matrix: 政策干预量 (T x N_FEAT)
        T: 规划年数
        df_panel_data: 面板数据
        bounds_low: 下界
        bounds_high: 上界
        natural_growth: 自然增长率字典 {"gdp_growth_rate": float, "elderly_change_rate": float}
                     如果为None，则GDP和老年人口比例保持不变
    """
    if bounds_low is None:
        bounds_low = BOUNDS_LOW
    if bounds_high is None:
        bounds_high = BOUNDS_HIGH
    states = np.zeros((T + 1, N_FEAT))
    hales  = np.zeros(T + 1)
    states[0] = x0.copy()
    hales[0]  = predict_hale_cached(tuple(np.round(x0, 12)), id(MODEL))
    
    # GDP在特征中的索引是3，老年人口比例索引是7
    GDP_IDX = FEATURE_COLS.index("gdp_per_capita_ppp")
    ELDERLY_IDX = FEATURE_COLS.index("elderly_share")
    
    for t in range(T):
        # 计算干预后的状态
        next_state = states[t] + delta_matrix[t]
        
        # 如果提供了自然增长率，应用GDP的自然增长
        if natural_growth is not None:
            gdp_growth = natural_growth.get("gdp_growth_rate", 0.0)
            if gdp_growth != 0:
                # GDP按复利增长
                next_state[GDP_IDX] = next_state[GDP_IDX] * (1 + gdp_growth)
        
        # 限制在边界内
        states[t+1] = np.clip(next_state, bounds_low, bounds_high)
        hales[t+1]  = predict_hale_cached(tuple(np.round(states[t+1], 12)), id(MODEL))
    return states, hales


class _TimeoutFlag:
    def __init__(self):
        self.triggered = False


def minimize_with_timeout(fun, x0, bounds, constraints, options, timeout_sec):
    flag = _TimeoutFlag()
    best = [x0.copy(), fun(x0)]
    lock = threading.Lock()

    def wrapped_fun(z):
        if flag.triggered:
            raise RuntimeError("__timeout__")
        val = fun(z)
        with lock:
            if val < best[1]:
                best[0] = z.copy()
                best[1] = val
        return val

    result_container = [None]

    def worker():
        try:
            result_container[0] = minimize(
                wrapped_fun, x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options=options
            )
        except RuntimeError:
            pass

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)

    timer = threading.Timer(timeout_sec, lambda: setattr(flag, "triggered", True))
    if not t.is_alive():
        timer.cancel()
    else:
        timer.start()

    if result_container[0] is not None:
        return result_container[0], False
    else:
        class _FakeResult:
            def __init__(self, x, f):
                self.x = x; self.fun = f; self.success = False
        return _FakeResult(best[0], best[1]), True


def optimize_policy_cli(country_name, T, Hale_input, year_start, df_panel_data, df_cluster_data,
                        custom_max_annual=None, progress_callback=None,
                        group_cost_weights=None, country_natural_growth=None,
                        custom_natural_growth=None):
    """
    优化政策

    Args:
        country_name: 国家名称
        T: 优化年数
        Hale_input: 用户输入的 HALE 目标值
        year_start: 起始年份
        df_panel_data: 面板数据
        df_cluster_data: 聚类数据
        custom_max_annual: 自定义每个指标的年改善上限（dict或None）
            格式: {"pm25_exposure": -3.0, "safe_water": 5.0, ...}
            None时使用默认的 MAX_ANNUAL
        progress_callback: 进度回调函数
        group_cost_weights: 分组成本权重字典 {"Low-Health": np.array([...]), ...}
            None时使用默认的 COST_WEIGHTS
        country_natural_growth: 国家自然增长率字典 {"country": {"gdp_growth_rate": float, ...}}
            None时使用默认值
        custom_natural_growth: 用户自定义的自然增长率 {"gdp_growth_rate": float, "elderly_change_rate": float}
            如果不为None，则覆盖country_natural_growth中的值
    """
    # 保存用户原始输入的目标
    Hale_input = float(Hale_input)
    Hale_target = Hale_input  # 初始化为用户输入值，后面可能调整
    
    x0, hale0, actual_year = get_initial_state(df_panel_data, country_name, year_start)
    hale_pred0 = predict_hale_cached(tuple(np.round(x0, 12)), id(MODEL))

    cluster_row = df_cluster_data[df_cluster_data["country"] == country_name]
    ctype = cluster_row["gmm_type"].values[0] if len(cluster_row) > 0 else "Mid-Health"

    # 干预方向: -1=降低(PM2.5), 0=不干预, 1=提高(其他)
    direction = np.array([-1, 1, 1, 0, 1, 1, 1, 0, 1])

    # 确定每个指标的年改善上限（提前定义，供hale_max_possible计算使用）
    if custom_max_annual is None:
        max_annual = MAX_ANNUAL.copy()
    else:
        max_annual = MAX_ANNUAL.copy()
        for feat, val in custom_max_annual.items():
            if feat in FEATURE_COLS:
                idx = FEATURE_COLS.index(feat)
                max_annual[idx] = val

    # 计算国家感知bounds（PM2.5不能低于当前值，防止模型倒U型陷阱）
    bnds_low, bnds_high = make_country_aware_bounds(x0)

    # 获取自然增长率（用户自定义 > 国家历史数据 > 默认值）
    if custom_natural_growth is not None:
        natural_growth = custom_natural_growth.copy()
    elif country_natural_growth is not None:
        natural_growth = get_country_natural_growth(country_name, country_natural_growth)
    else:
        natural_growth = None

    # 计算 HALE 极限值（考虑干预上限后的可达上限）
    # 方法：模拟10年后所有指标在干预下的最大可达状态
    T_ESTIMATE = 10
    x_max = x0.copy()
    for i, feat in enumerate(FEATURE_COLS):
        if feat == 'gdp_per_capita_ppp':
            gdp_gr = natural_growth.get("gdp_growth_rate", 0) if natural_growth else 0
            x_max[i] = x0[i] * ((1 + gdp_gr) ** T_ESTIMATE)
        elif feat in ['elderly_share']:
            elderly_ch = natural_growth.get("elderly_change_rate", 0) if natural_growth else 0
            x_max[i] = x0[i] + elderly_ch * T_ESTIMATE
        elif feat == "pm25_exposure":
            max_reduction = abs(max_annual[i]) * T_ESTIMATE
            x_max[i] = max(0.0, x0[i] - max_reduction)
        elif direction[i] == 1:
            max_increase = abs(max_annual[i]) * T_ESTIMATE
            x_max[i] = min(bnds_high[i], x0[i] + max_increase)
        else:
            x_max[i] = bnds_high[i]

    # 清缓存计算极限
    predict_hale_cached.cache_clear()
    hale_max_possible = predict_hale_cached(tuple(np.round(x_max, 12)), id(MODEL))
    predict_hale_cached.cache_clear()

    # 如果目标超过模型极限，自动将目标设为可达上限
    if Hale_target > hale_max_possible + 0.1:
        Hale_target = min(hale_max_possible + 0.5, Hale_target)

    gap                 = Hale_target - hale_pred0
    # 惩罚权重 = 确保惩罚项远大于政策成本项
    penalty_weight      = max(5000.0, abs(gap) * 2000)

    # 根据国家所属组选择成本权重
    if group_cost_weights is not None and ctype in group_cost_weights:
        cost_weights = group_cost_weights[ctype]
    else:
        cost_weights = COST_WEIGHTS

    # 使用国家感知bounds
    # direction: -1=降低(PM2.5), 0=不干预, 1=提高(其他)
    direction_local = np.array([-1, 1, 1, 0, 1, 1, 1, 0, 1])
    var_bounds = []
    for t in range(T):
        for i, feat in enumerate(FEATURE_COLS):
            if feat in ["gdp_per_capita_ppp", "elderly_share"]:
                # 不干预的指标，delta 固定为 0
                var_bounds.append((0, 0))
            elif feat == "pm25_exposure":
                # PM2.5: direction=-1 表示需要降低
                # delta 范围：每年最多降低 abs(max_annual)，最少不变 (0)
                lo = -abs(max_annual[i])  # 最多降低这么多（负值）
                hi = 0.0  # 不能增加（delta 不能为正）
                var_bounds.append((lo, hi))
            else:
                # 其他正向指标：direction=1 时，delta 只能为正（增加）
                if direction_local[i] == 1:
                    lo = 0.0  # 至少不减少
                    hi = abs(max_annual[i])  # 最多增加这么多
                else:
                    lo = -abs(max_annual[i])  # 允许减少
                    hi = abs(max_annual[i])   # 允许增加
                var_bounds.append((lo, hi))

    def total_cost(z):
        delta = z.reshape(T, N_FEAT)
        cost  = 0.0
        for t in range(T):
            disc = 1.0 / (1 + DISCOUNT_RATE) ** t
            cost += disc * np.sum(cost_weights * np.abs(delta[t]))
        _, hales = simulate_trajectory(x0, delta, T, df_panel_data, bnds_low, bnds_high, natural_growth)
        shortfall = max(0.0, Hale_target - hales[-1])
        cost += penalty_weight * shortfall ** 2
        return cost

    def final_hale_con(z):
        delta = z.reshape(T, N_FEAT)
        _, hales = simulate_trajectory(x0, delta, T, df_panel_data, bnds_low, bnds_high, natural_growth)
        return hales[-1] - Hale_target

    constraints = [{"type": "ineq", "fun": final_hale_con}]
    options = {"maxiter": 500, "ftol": 1e-4, "disp": False}

    best_result = None
    best_hale   = -np.inf
    strategies   = ["uniform", "frontload"] + ["random"] * 3

    for i, strategy in enumerate(strategies):
        np.random.seed(i * 13)

        if strategy == "uniform":
            # 使用更柔和的初始步长（0.15倍），避免步长过大导致优化器跳过最优解
            per_year = max_annual * 0.15 * direction
            delta    = np.tile(per_year, (T, 1))
        elif strategy == "frontload":
            high = max_annual * 0.2 * direction
            low  = max_annual * 0.05 * direction
            delta = np.vstack([np.tile(high, (T // 2, 1)), np.tile(low, (T - T // 2, 1))])
        else:
            delta = np.random.uniform(0, 1, (T, N_FEAT)) * max_annual * direction * 0.2

        z_init = delta.flatten()
        res, _ = minimize_with_timeout(total_cost, z_init, var_bounds, constraints, options, RESTART_TIMEOUT)

        delta_try, hales_try = simulate_trajectory(x0, res.x.reshape(T, N_FEAT), T, df_panel_data, bnds_low, bnds_high, natural_growth)
        final_h = hales_try[-1]

        if final_h > best_hale:
            best_hale   = final_h
            best_result = res

        if progress_callback:
            progress_callback(i + 1, len(strategies), strategy, final_h, res.fun, res.success)

    if best_result is None:
        raise RuntimeError(f"{country_name}: 所有 restart 均失败")

    delta_opt       = best_result.x.reshape(T, N_FEAT)
    states_opt, hales_opt   = simulate_trajectory(x0, delta_opt, T, df_panel_data, bnds_low, bnds_high, natural_growth)
    # 基线模拟也使用自然增长率
    _, hales_base = simulate_trajectory(x0, np.zeros((T, N_FEAT)), T, df_panel_data, bnds_low, bnds_high, natural_growth)

    cost_by_feature = np.zeros(N_FEAT)
    for t in range(T):
        disc = 1.0 / (1 + DISCOUNT_RATE) ** t
        cost_by_feature += disc * cost_weights * np.abs(delta_opt[t])

    return {
        "country":             country_name,
        "cluster_type":        ctype,
        "T":                   T,
        "hale_target":         Hale_target,       # 实际使用的目标（可能被调整过）
        "hale_target_input":   Hale_input,       # 用户原始输入的目标
        "hale_initial":        hale0,
        "hale_pred_initial":   hale_pred0,
        "hale_final":          hales_opt[-1],
        "hale_baseline_final": hales_base[-1],
        "success":             best_result.success,
        "delta_opt":           delta_opt,
        "states_opt":          states_opt,
        "hales_opt":           hales_opt,
        "hales_base":         hales_base,
        "cost_by_feature":    cost_by_feature,
        "total_cost":         cost_by_feature.sum(),
        "years":              np.arange(actual_year, actual_year + T + 1),
        "x0":                 x0,
        "cost_weights_used":  cost_weights,  # 记录使用的成本权重
        "natural_growth":     natural_growth,  # 记录使用的自然增长率
    }


# ─────────────────────────────────────────────
# 全局模型加载
# ─────────────────────────────────────────────
if "MODEL" not in st.session_state:
    try:
        st.session_state.MODEL = load_model()
        st.session_state.PANEL  = load_panel()
        st.session_state.CLUSTER = load_cluster()
        st.session_state.SUMMARY = load_summary()
        st.session_state["data_loaded"] = True
    except FileNotFoundError as e:
        st.session_state["data_loaded"] = False
        st.session_state["load_error"] = str(e)

MODEL = st.session_state.get("MODEL")
PANEL = st.session_state.get("PANEL")
CLUSTER = st.session_state.get("CLUSTER")
SUMMARY = st.session_state.get("SUMMARY")

DATA_LOADED = st.session_state.get("data_loaded", False)

# ─────────────────────────────────────────────
# 预计算分组成本权重（数据驱动）
# ─────────────────────────────────────────────
GROUP_COST_WEIGHTS = None
COUNTRY_NATURAL_GROWTH = None
if DATA_LOADED and PANEL is not None and CLUSTER is not None:
    # 干预方向: -1=降低(PM2.5), 0=不干预(GDP, elderly), 1=提高(其他)
    direction = np.array([-1, 1, 1, 0, 1, 1, 1, 0, 1])
    GROUP_COST_WEIGHTS = compute_group_cost_weights(PANEL, CLUSTER, FEATURE_COLS, direction)
    # 计算每个国家的自然增长率
    COUNTRY_NATURAL_GROWTH = compute_country_natural_growth(PANEL)

# ─────────────────────────────────────────────
# 侧边栏导航
# ─────────────────────────────────────────────
st.sidebar.markdown("""
<div class="sidebar-header">
    <h2>🌍 健康决策系统</h2>
</div>
""", unsafe_allow_html=True)

# 定义导航选项和图标
nav_options = [
    ("🏠", "项目概览", "project"),
    ("🌡", "国家画像", "profile"),
    ("🎯", "健康优化", "optimize"),
    ("📊", "结果汇总", "summary")
]

# 创建自定义导航样式
nav_style = """
<style>
.nav-button {
    display: flex;
    align-items: center;
    padding: 14px 20px;
    margin: 8px 0;
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    color: white;
    font-size: 1rem;
    cursor: pointer;
    transition: all 0.3s ease;
    border: 1px solid transparent;
}
.nav-button:hover {
    background: rgba(16, 185, 129, 0.3);
    transform: translateX(6px);
}
.nav-button.active {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%);
    border-color: rgba(255,255,255,0.2);
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
}
.nav-icon {
    font-size: 1.3rem;
    margin-right: 12px;
}
.nav-text {
    font-weight: 500;
}
</style>
"""
st.sidebar.markdown(nav_style, unsafe_allow_html=True)

# 导航选项
page_options = ["🏠 项目概览", "🌡 国家画像", "🎯 健康优化", "📊 结果汇总"]
page_icons = ["🏠", "🌡", "🎯", "📊"]

# 使用选中的选项显示当前页面
selected_idx = 0
page = st.sidebar.radio(
    "导航",
    page_options,
    index=selected_idx,
    label_visibility="collapsed"
)

# 底部信息
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; padding: 15px; background: rgba(255,255,255,0.05); border-radius: 12px; margin-top: 20px;">
    <div style="font-size: 0.9rem; color: rgba(255,255,255,0.8);">
        <b>健康决策支持系统</b><br>
        <span style="font-size: 0.8rem;">基于 XGBoost + 多期优化</span>
    </div>
    <div style="font-size: 0.75rem; color: rgba(255,255,255,0.5); margin-top: 8px;">
        覆盖 146 个国家 | 2000-2021
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 页面 1：项目概览
# ─────────────────────────────────────────────
if page == "🏠 项目概览":
    # 标题区域
    st.markdown("""
    <div style="text-align: center; padding: 30px 0;">
        <h1 class="main-title">🌍 健康决策支持系统</h1>
        <p class="sub-title">基于 <b>XGBoost</b> 健康预测模型 + <b>多期优化算法</b>，为全球各国提供健康改善路径建议</p>
    </div>
    """, unsafe_allow_html=True)

    if not DATA_LOADED:
        st.error(f"数据加载失败：{st.session_state.get('load_error', '未知错误')}")
        st.stop()

    # 核心指标卡片
    st.markdown('<h3 class="section-header">📌 数据集总览</h3>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <div style="font-size: 2.5rem; color: #10b981;">🌐</div>
            <div style="font-size: 2rem; font-weight: 700; color: #047857;">146</div>
            <div style="color: #718096; font-size: 0.95rem;">覆盖国家</div>
            <div style="color: #a0aec0; font-size: 0.8rem; margin-top: 5px;">全球范围</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <div style="font-size: 2.5rem; color: #0284c7;">📅</div>
            <div style="font-size: 2rem; font-weight: 700; color: #047857;">2000-2021</div>
            <div style="color: #718096; font-size: 0.95rem;">时间跨度</div>
            <div style="color: #a0aec0; font-size: 0.8rem; margin-top: 5px;">22年数据</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <div style="font-size: 2.5rem; color: #10b981;">📊</div>
            <div style="font-size: 2rem; font-weight: 700; color: #047857;">3,177</div>
            <div style="color: #718096; font-size: 0.95rem;">数据记录</div>
            <div style="color: #a0aec0; font-size: 0.8rem; margin-top: 5px;">条记录</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <div style="font-size: 2.5rem; color: #0284c7;">🎯</div>
            <div style="font-size: 2rem; font-weight: 700; color: #047857;">3 类</div>
            <div style="color: #718096; font-size: 0.95rem;">聚类分组</div>
            <div style="color: #a0aec0; font-size: 0.8rem; margin-top: 5px;">Low/Mid/High</div>
        </div>
        """, unsafe_allow_html=True)

    # 模型性能
    st.markdown("""
    <hr class="custom-divider">
    <h3 class="section-header">🤖 XGBoost 模型性能</h3>
    """, unsafe_allow_html=True)

    perf_df = pd.read_csv(os.path.join(OUT_DIR, "model_performance_v2.csv"), header=None, names=["metric", "value"])
    perf_df = perf_df.set_index("metric")["value"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        mae_val = float(perf_df.get('test_mae', 0))
        st.markdown(f"""
        <div class="metric-card" style="text-align: center; border-left: 4px solid #10b981;">
            <div style="font-size: 1.5rem; color: #10b981;">📉</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #047857;">{mae_val:.2f}</div>
            <div style="color: #718096; font-size: 0.9rem;">测试集 MAE</div>
            <div style="color: #a0aec0; font-size: 0.8rem;">岁</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        rmse_val = float(perf_df.get('test_rmse', 0))
        st.markdown(f"""
        <div class="metric-card" style="text-align: center; border-left: 4px solid #10b981;">
            <div style="font-size: 1.5rem; color: #10b981;">📊</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #047857;">{rmse_val:.2f}</div>
            <div style="color: #718096; font-size: 0.9rem;">测试集 RMSE</div>
            <div style="color: #a0aec0; font-size: 0.8rem;">岁</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        r2_val = float(perf_df.get('test_r2', 0))
        st.markdown(f"""
        <div class="metric-card" style="text-align: center; border-left: 4px solid #10b981;">
            <div style="font-size: 1.5rem; color: #10b981;">🎯</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #047857;">{r2_val:.3f}</div>
            <div style="color: #718096; font-size: 0.9rem;">测试集 R²</div>
            <div style="color: #a0aec0; font-size: 0.8rem;">拟合优度</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        cv_r2 = float(perf_df.get('cv_r2_mean', 0))
        st.markdown(f"""
        <div class="metric-card" style="text-align: center; border-left: 4px solid #0284c7;">
            <div style="font-size: 1.5rem; color: #0284c7;">🔄</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #047857;">{cv_r2:.3f}</div>
            <div style="color: #718096; font-size: 0.9rem;">交叉验证 R²</div>
            <div style="color: #a0aec0; font-size: 0.8rem;">稳健性</div>
        </div>
        """, unsafe_allow_html=True)

    # 聚类分布
    st.markdown("""
    <hr class="custom-divider">
    <h3 class="section-header">🗺 聚类分布</h3>
    """, unsafe_allow_html=True)

    cluster_counts = CLUSTER["gmm_type"].value_counts()
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("**📍 分类统计**")
        for ctype in ["Low-Health", "Mid-Health", "High-Health"]:
            cnt = cluster_counts.get(ctype, 0)
            pct = cnt / cluster_counts.sum() * 100
            color = CLUSTER_PALETTE.get(ctype, "#888")
            st.markdown(f"""
            <div style="display:flex; align-items:center; margin:12px 0; padding: 10px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 10px;">
                <div style="width:16px;height:16px;background:{color};border-radius:50%;margin-right:12px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);"></div>
                <span style="font-size:14px;font-weight:500;">{ctype}</span>
                <span style="margin-left:auto;font-weight:700;color:#2d3748;">{cnt} 国</span>
                <span style="color:#718096;font-size:0.85rem;margin-left:8px;">({pct:.1f}%)</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        colors = [CLUSTER_PALETTE.get(c, "#888") for c in cluster_counts.index]
        bars = ax.barh(cluster_counts.index, cluster_counts.values, color=colors, edgecolor="white", height=0.6)
        for bar, val in zip(bars, cluster_counts.values):
            ax.text(val + 1, bar.get_y() + bar.get_height() / 2, f" {val}国",
                    va="center", fontsize=11, fontweight='bold')
        ax.set_xlabel("国家数量", fontsize=11)
        ax.set_title("国家健康聚类分布", fontsize=13, fontweight='bold', pad=15)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0, max(cluster_counts.values) * 1.2)
        ax.grid(axis="x", alpha=0.3)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#f8f9fa')
        st.pyplot(fig)

    # 特征重要性（如果有图）
    st.markdown("""
    <hr class="custom-divider">
    <h3 class="section-header">📈 模型分析</h3>
    """, unsafe_allow_html=True)

    imp_path = os.path.join(FIG_PRED_DIR, "fig3_feature_importance.png")
    pred_path = os.path.join(FIG_PRED_DIR, "fig1_pred_vs_actual.png")
    radar_path = os.path.join(FIG_CLS_DIR, "fig5_radar_k3.png")
    scatter_path = os.path.join(FIG_CLS_DIR, "fig2_scatter_k3.png")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("**🎯 特征重要性**")
        if os.path.exists(imp_path):
            from PIL import Image
            img = Image.open(imp_path)
            st.image(img, use_container_width=True)
        else:
            st.info("特征重要性图暂未生成")
        st.markdown('</div>', unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("**📊 预测 vs 实际**")
        if os.path.exists(pred_path):
            from PIL import Image
            img = Image.open(pred_path)
            st.image(img, use_container_width=True)
        else:
            st.info("预测对比图暂未生成")
        st.markdown('</div>', unsafe_allow_html=True)

    # 聚类图
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("**🗺️ 聚类雷达图**")
        if os.path.exists(radar_path):
            st.image(Image.open(radar_path), use_container_width=True)
        else:
            st.info("聚类雷达图暂未生成")
        st.markdown('</div>', unsafe_allow_html=True)
    with col_d:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("**🧬 聚类散点图**")
        if os.path.exists(scatter_path):
            st.image(Image.open(scatter_path), use_container_width=True)
        else:
            st.info("聚类散点图暂未生成")
        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 页面 2：国家画像
# ─────────────────────────────────────────────
elif page == "🌡 国家画像":
    # 标题区域
    st.markdown("""
    <div style="text-align: center; padding: 30px 0;">
        <h1 class="main-title">🌡 国家健康画像</h1>
        <p class="sub-title">选择国家，查看其健康指标分布、与同类国家的对比、以及历史趋势</p>
    </div>
    """, unsafe_allow_html=True)

    if not DATA_LOADED:
        st.error("数据加载失败"); st.stop()

    all_countries = sorted(PANEL["country"].unique().tolist())

    # 搜索框区域
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    col_search, col_country = st.columns([2, 3])
    with col_search:
        search = st.text_input("🔍 搜索国家（输入英文名）", "", placeholder="输入国家名称搜索...").strip()
    if search:
        candidates = [c for c in all_countries if search.lower() in c.lower()]
        if candidates:
            default_idx = all_countries.index(candidates[0]) if candidates[0] in all_countries else 0
        else:
            st.warning(f"未找到包含「{search}」的国家")
            default_idx = 0
    else:
        default_idx = 0
    with col_country:
        country = st.selectbox("选择国家", all_countries, index=default_idx)
    st.markdown('</div>', unsafe_allow_html=True)

    # 基础信息
    row_c = CLUSTER[CLUSTER["country"] == country]
    ctype = row_c["gmm_type"].values[0] if len(row_c) > 0 else "Unknown"
    hale_mean = row_c["HALE_mean"].values[0] if len(row_c) > 0 else None
    color = CLUSTER_PALETTE.get(ctype, "#888")

    # 国家信息卡片
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {color} 0%, {color}88 100%); 
                color: white; border-radius: 16px; padding: 24px; margin: 15px 0; 
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h2 style="margin: 0; font-size: 1.8rem;">{country}</h2>
                <div style="font-size: 1rem; opacity: 0.9; margin-top: 5px;">
                    <span style="background: rgba(255,255,255,0.2); padding: 4px 12px; border-radius: 15px;">{ctype}</span>
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 2.5rem; font-weight: 700;">{f"{hale_mean:.1f}" if hale_mean is not None else "N/A"}</div>
                <div style="font-size: 0.95rem; opacity: 0.9;">平均 HALE（岁）</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 三列图：雷达 / 特征条形 / HALE 趋势
    st.markdown('<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0;">', unsafe_allow_html=True)

    col_radar, col_bar, col_trend = st.columns(3)

    with col_radar:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("**📡 健康指标雷达图**")
        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor('white')
        features_display = ["pm25_exposure", "safe_water", "basic_sanitation",
                           "physicians_per_1000", "health_exp_gdp", "G_norm"]
        feat_labels_disp = [FEATURE_LABELS[f] for f in features_display]
        country_feats = row_c[features_display].values.flatten() if len(row_c) > 0 else [0] * len(features_display)
        global_mean  = CLUSTER[features_display].mean().values
        angles = np.linspace(0, 2 * np.pi, len(features_display), endpoint=False).tolist()
        angles += angles[:1]
        c_vals  = (np.array(country_feats) / 100).tolist() if max(country_feats) > 1 else country_feats.tolist()
        c_vals  += c_vals[:1]
        g_vals  = (global_mean / 100).tolist() if max(global_mean) > 1 else global_mean.tolist()
        g_vals  += g_vals[:1]
        ax.plot(angles, c_vals, "o-", color=color, lw=2.5, label=country, markersize=6)
        ax.fill(angles, c_vals, alpha=0.3, color=color)
        ax.plot(angles, g_vals, "--", color="gray", lw=2, label="Global Mean", alpha=0.7)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feat_labels_disp, size=8, fontweight='bold')
        ax.set_ylim(0, 1) if max(country_feats) > 1 else ax.set_ylim(0, max(max(country_feats)*1.2, 1))
        ax.set_title(f"{country}\n{ctype}", size=12, pad=20, fontweight='bold')
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
        ax.grid(color='#e0e0e0', linestyle='--', linewidth=0.8)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_bar:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("**📊 与全球均值对比**")
        feat_comp = ["pm25_exposure", "safe_water", "basic_sanitation",
                     "physicians_per_1000", "beds_per_1000", "health_exp_gdp", "G_norm"]
        feat_lbl  = [FEATURE_LABELS[f] for f in feat_comp]
        c_vals2    = row_c[feat_comp].values.flatten() if len(row_c) > 0 else [0] * len(feat_comp)
        g_vals2    = CLUSTER[feat_comp].mean().values
        x = np.arange(len(feat_comp))
        fig, ax = plt.subplots(figsize=(5.5, 5))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#f8f9fa')
        w = 0.35
        ax.barh(x - w/2, c_vals2, w, label=country, color=color, alpha=0.85, edgecolor='white')
        ax.barh(x + w/2, g_vals2, w, label="Global Mean", color="lightgray", alpha=0.85, edgecolor='white')
        ax.set_yticks(x)
        ax.set_yticklabels(feat_lbl, fontsize=9, fontweight='500')
        ax.set_xlabel("Value", fontsize=10)
        ax.legend(fontsize=9, loc='lower right')
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.3)
        ax.set_title(f"{country} vs Global Mean", fontsize=11, fontweight='bold', pad=10)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_trend:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("**📈 HALE 历史趋势**")
        sub = PANEL[PANEL["country"] == country].sort_values("year")
        fig, ax = plt.subplots(figsize=(5.5, 5))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#f8f9fa')
        ax.plot(sub["year"], sub["HALE"], "o-", color=color, lw=2.5, ms=6, label=country)
        ax.fill_between(sub["year"], sub["HALE"], alpha=0.2, color=color)
        ax.axhline(sub["HALE"].mean(), color=color, ls="--", lw=1.5, alpha=0.6, label=f"均值 {sub['HALE'].mean():.1f}")
        ax.set_xlabel("年份", fontsize=10)
        ax.set_ylabel("HALE (岁)", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title("HALE 历史变化趋势", fontsize=11, fontweight='bold', pad=10)
        # 添加起始和结束标注
        if len(sub) > 0:
            ax.annotate(f'{sub["HALE"].iloc[0]:.1f}', 
                       xy=(sub["year"].iloc[0], sub["HALE"].iloc[0]),
                       xytext=(5, 10), textcoords='offset points', fontsize=9, color=color)
            ax.annotate(f'{sub["HALE"].iloc[-1]:.1f}',
                       xy=(sub["year"].iloc[-1], sub["HALE"].iloc[-1]),
                       xytext=(5, 10), textcoords='offset points', fontsize=9, color=color)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # 同类国家对比
    st.markdown("""
    <hr class="custom-divider">
    <h3 class="section-header">🔗 与同类国家对比（{}）</h3>
    """.format(ctype), unsafe_allow_html=True)

    same_cluster = CLUSTER[CLUSTER["gmm_type"] == ctype].sort_values("HALE_mean", ascending=False)
    same_cluster = same_cluster[same_cluster["country"] != country].head(10)
    comp_cols = ["country", "HALE_mean"] + FEATURE_COLS[:6]
    comp_display = same_cluster[[c for c in comp_cols if c in same_cluster.columns]].head(10)
    comp_display = comp_display.round(2)

    st.markdown('<div class="data-table">', unsafe_allow_html=True)
    st.dataframe(comp_display, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 时序详情
    st.markdown("""
    <hr class="custom-divider">
    <h3 class="section-header">📋 历史数据明细</h3>
    """, unsafe_allow_html=True)

    st.markdown('<div class="data-table">', unsafe_allow_html=True)
    st.dataframe(sub[["year", "HALE"] + FEATURE_COLS[:6]].rename(columns={**FEATURE_LABELS, **{"year": "年份", "HALE": "HALE"}}).round(2),
                 use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 页面 3：健康优化
# ─────────────────────────────────────────────
elif page == "🎯 健康优化":
    st.title("🎯 健康水平提升优化")
    st.markdown("选择国家、设定目标，系统将基于 XGBoost 模型预测 + 多期优化算法，生成最优政策路径图。")

    if not DATA_LOADED:
        st.error("数据加载失败"); st.stop()

    all_countries = sorted(PANEL["country"].unique().tolist())

    # 交互输入区
    with st.container():
        st.markdown("### 📥 输入参数")
        inp1, inp2, inp3 = st.columns([2, 1, 1])

        with inp1:
            search2 = st.text_input("🔍 搜索国家", "", label_visibility="collapsed",
                                   placeholder="输入国家英文名搜索...")
            if search2:
                matches = [c for c in all_countries if search2.lower() in c.lower()]
                if matches:
                    default_i = all_countries.index(matches[0])
                else:
                    st.warning(f"未找到「{search2}」")
                    default_i = 0
            else:
                default_i = all_countries.index("Kenya") if "Kenya" in all_countries else 0
            country_sel = st.selectbox("选择国家", all_countries, index=default_i, label_visibility="collapsed")

        with inp2:
            T_input = st.number_input("规划年数", min_value=1, max_value=30, value=5, step=1)

        with inp3:
            n_restarts = st.number_input("重启次数", min_value=1, max_value=20, value=5, step=1)

    # 获取当前国家数据
    x0_sel, hale0_sel, year_sel = get_initial_state(PANEL, country_sel, 2019)
    hale_pred_sel = predict_hale_cached(tuple(np.round(x0_sel, 12)), id(MODEL))

    # 计算该国实际可达的最大 HALE（使用国家感知bounds）
    bnds_low_sel, bnds_high_sel = make_country_aware_bounds(x0_sel)
    x_max_sel = x0_sel.copy()
    direction_local = np.array([-1, 1, 1, 0, 1, 1, 1, 0, 1])
    for i, feat in enumerate(FEATURE_COLS):
        if feat in ['gdp_per_capita_ppp', 'elderly_share']:
            continue
        elif direction_local[i] == -1:
            x_max_sel[i] = bnds_low_sel[i]
        else:
            x_max_sel[i] = bnds_high_sel[i]
    predict_hale_cached.cache_clear()
    hale_max_sel = predict_hale_cached(tuple(np.round(x_max_sel, 12)), id(MODEL))
    predict_hale_cached.cache_clear()

    row_sel = CLUSTER[CLUSTER["country"] == country_sel]
    ctype_sel = row_sel["gmm_type"].values[0] if len(row_sel) > 0 else "Mid-Health"
    # 计算滑块范围
    # 注意：模型预测的理论上限(hale_max_sel)可能很小，我们使用更合理的范围
    # 1. 下限：当前预测值 - 1（允许略低于预测）
    # 2. 上限：基于集群类型的经验上限，允许用户设置更高的改善目标
    cluster_hale_max = {"Low-Health": 70.0, "Mid-Health": 80.0, "High-Health": 85.0}
    slider_min = float(max(hale_pred_sel - 1.0, 0))
    # 使用模型预测上限和集群类型经验上限中的较大值，确保滑块有足够的范围
    empirical_max = cluster_hale_max.get(ctype_sel, 75.0)
    slider_max = float(max(hale_max_sel, empirical_max))
    # 默认放在滑块范围的中间偏上位置
    default_target = round((slider_min + slider_max) / 2.0, 1)

    # 构建当前指标数据字典（用于大模型推荐）
    current_data = {}
    for feat, label in FEATURE_LABELS.items():
        idx = FEATURE_COLS.index(feat)
        current_data[feat] = {"value": float(x0_sel[idx]), "label": label}

    # ─────────────────────────────────────────────
    # 大模型辅助：推荐约束（需要用户确认才应用）
    # ─────────────────────────────────────────────
    llm_recommended = {}
    use_llm_recommendation = False
    
    # 初始化session_state存储推荐值和用户自定义值
    if "llm_recommendation_applied" not in st.session_state:
        st.session_state.llm_recommendation_applied = False
    if "user_adjusted_constraints" not in st.session_state:
        st.session_state.user_adjusted_constraints = {}
    if "pending_llm_recommendation" not in st.session_state:
        st.session_state.pending_llm_recommendation = {}
    
    # 检测国家是否变化，如果变化则重置状态
    if "last_country" not in st.session_state or st.session_state.get("last_country") != country_sel:
        st.session_state.last_country = country_sel
        st.session_state.llm_recommendation_applied = False
        st.session_state.user_adjusted_constraints = {}
        st.session_state.pending_llm_recommendation = {}
    
    # 如果还没有获取推荐（无论是用户请求还是初始状态），获取大模型推荐
    if not st.session_state.pending_llm_recommendation:
        if LLM_CONFIG["enabled"] and LLM_CONFIG["api_key"]:
            with st.spinner("🤖 大模型正在分析该国情况并推荐约束..."):
                llm_recommended = call_llm_recommend_constraints(
                    country_sel, current_data, FEATURE_LABELS, ctype_sel
                )
            if llm_recommended:
                st.session_state.pending_llm_recommendation = llm_recommended
                st.info(f"💡 大模型已为 {country_sel} 生成推荐，请查看下方约束调整区域，确认后点击「应用大模型推荐」按钮")
            else:
                st.warning("⚠️ 大模型调用未返回有效结果，将使用默认约束")
    else:
        llm_recommended = st.session_state.pending_llm_recommendation

    # 大模型推荐确认按钮
    if LLM_CONFIG["enabled"] and LLM_CONFIG["api_key"] and llm_recommended:
        if st.button("🤖 应用大模型推荐", type="secondary", use_container_width=False):
            st.session_state.user_adjusted_constraints = llm_recommended.copy()
            st.session_state.llm_recommendation_applied = True
            st.success("✅ 已应用大模型推荐！你可以继续手动调整。")
            st.rerun()
    
    # 检查是否有已经应用过的推荐
    if st.session_state.llm_recommendation_applied and not st.session_state.user_adjusted_constraints:
        st.session_state.user_adjusted_constraints = llm_recommended.copy()

    # ─────────────────────────────────────────────
    # 获取该国的自然增长率（用于显示）
    # ─────────────────────────────────────────────
    country_nat = {}
    if COUNTRY_NATURAL_GROWTH is not None:
        country_nat = COUNTRY_NATURAL_GROWTH.get(country_sel, {})
    
    # 显示自然增长率信息卡片
    gdp_growth_display = country_nat.get("gdp_growth_rate", 0.03)
    elderly_change_display = country_nat.get("elderly_change_rate", 0.3)
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
                border-radius: 16px; padding: 16px; margin: 15px 0;
                border-left: 4px solid #d97706;">
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <span style="font-size: 1.5rem; margin-right: 10px;">📊</span>
            <span style="font-weight: 600; color: #92400e;">背景趋势（基于历史数据计算，不建议调整）</span>
        </div>
        <div style="display: flex; gap: 30px; flex-wrap: wrap;">
            <div style="background: white; padding: 12px 20px; border-radius: 10px; min-width: 200px;">
                <div style="font-size: 0.85rem; color: #718096;">GDP 年均增长率</div>
                <div style="font-size: 1.4rem; font-weight: 700; color: #059669;">{gdp_growth_display*100:.2f}%</div>
                <div style="font-size: 0.75rem; color: #a0aec0; margin-top: 4px;">
                    基于 {country_sel} 历史数据计算
                </div>
            </div>
            <div style="background: white; padding: 12px 20px; border-radius: 10px; min-width: 200px;">
                <div style="font-size: 0.85rem; color: #718096;">老年人口比例年均变化</div>
                <div style="font-size: 1.4rem; font-weight: 700; color: #059669;">{elderly_change_display:+.2f} 个点</div>
                <div style="font-size: 0.75rem; color: #a0aec0; margin-top: 4px;">
                    百分点/年，正值表示老龄化加深
                </div>
            </div>
        </div>
        <div style="margin-top: 10px; font-size: 0.8rem; color: #92400e;">
            💡 这些值基于该国2000-2021年的历史数据计算，反映GDP自然增长和人口老龄化的背景趋势。
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ─────────────────────────────────────────────
    # 自然增长率微调区（可选，允许用户在小范围内调整）
    # ─────────────────────────────────────────────
    with st.expander("🔧 微调背景趋势（可选）", expanded=False):
        st.markdown("""
        <div class="warning-box">
            ⚠️ <b>谨慎使用：</b>这些参数基于历史数据计算，通常不需要调整。
            只有在有特殊假设（如经济危机、战争、人口政策变化）时才建议调整。
        </div>
        """, unsafe_allow_html=True)

        # 初始化 custom_natural_growth
        custom_natural_growth = {
            "gdp_growth_rate": gdp_growth_display,
            "elderly_change_rate": elderly_change_display
        }
        
        col_nat1, col_nat2 = st.columns(2)
        
        # GDP增长率微调
        with col_nat1:
            custom_gdp_growth = st.slider(
                "GDP 年均增长率 (%)",
                min_value=-10.0,
                max_value=30.0,
                value=float(gdp_growth_display * 100),
                step=0.1,
                format="%.1f",
                help=f"基于历史数据计算的默认值为 {gdp_growth_display*100:.2f}%，表示GDP每年平均增长的比例"
            )
        
        # 老年人口变化微调
        with col_nat2:
            custom_elderly_change = st.slider(
                "老年人口比例年均变化 (百分点/年)",
                min_value=-2.0,
                max_value=3.0,
                value=float(elderly_change_display),
                step=0.1,
                format="%.1f",
                help=f"基于历史数据计算的默认值为 {elderly_change_display:.2f}，正值表示老龄化加深"
            )
        
        # 更新用户自定义的自然增长率
        custom_natural_growth = {
            "gdp_growth_rate": custom_gdp_growth / 100.0,  # 转回小数
            "elderly_change_rate": custom_elderly_change
        }
        
        # 显示调整说明
        if abs(custom_gdp_growth - gdp_growth_display * 100) > 0.5 or abs(custom_elderly_change - elderly_change_display) > 0.1:
            st.info(f"📝 已将 GDP 增长率调整为 {custom_gdp_growth:.1f}%，老年人口变化调整为 {custom_elderly_change:.2f} 个点/年")

    color_sel = CLUSTER_PALETTE.get(ctype_sel, "#888")

    # ─────────────────────────────────────────────
    # 约束调整区
    # ─────────────────────────────────────────────
    with st.expander("⚙️ 高级选项：调整每指标年改善上限", expanded=False):
        st.markdown("""
        <div class="info-box">
            💡 <b>提示：</b>你可以设置每个健康指标每年的最大改善幅度。
            大模型会自动根据该国实际情况推荐合适的值，你也可以手动调整。
        </div>
        """, unsafe_allow_html=True)

        col_feat1, col_feat2, col_feat3 = st.columns(3)
        feat_cols = list(FEATURE_COLS)
        feat_labels_display = {
            "pm25_exposure": "PM2.5 暴露量",
            "safe_water": "安全饮水率",
            "basic_sanitation": "基础卫生率",
            "gdp_per_capita_ppp": "人均GDP",
            "physicians_per_1000": "医生密度",
            "beds_per_1000": "床位密度",
            "health_exp_gdp": "卫生支出占比",
            "elderly_share": "老年人口比例",
            "G_norm": "治理指数",
        }

        custom_max_annual = {}

        for i, feat in enumerate(feat_cols):
            col = [col_feat1, col_feat2, col_feat3][i % 3]

            # 获取大模型推荐值和说明
            llm_data = llm_recommended.get(feat, {})
            llm_val = llm_data.get("max_annual", None)
            llm_note = llm_data.get("note", "")
            
            # 优先使用用户已调整的值（存储在session_state中）
            if feat in st.session_state.user_adjusted_constraints:
                user_val = st.session_state.user_adjusted_constraints[feat]
                if isinstance(user_val, dict) and "max_annual" in user_val:
                    user_val = user_val["max_annual"]
                default_val = float(user_val) if user_val is not None else float(MAX_ANNUAL[i])
            elif llm_val is not None:
                default_val = float(llm_val)
            else:
                default_val = float(MAX_ANNUAL[i])
            
            # 获取大模型推荐的说明
            llm_note_text = llm_note if llm_note else "无"
            llm_help = f"{llm_val:.2f}" if llm_val is not None else "默认"
            
            # 使用统一的范围 [-10, 10]，确保所有滑块都能正常工作
            # 这样既能让用户有足够的调整空间，又能保证滑块可用
            min_val = -10.0
            max_val = 10.0
            
            # 确保默认值在范围内
            if default_val < min_val:
                default_val = min_val
            if default_val > max_val:
                default_val = max_val

            with col:
                new_val = st.slider(
                    f"{feat_labels_display.get(feat, feat)}",
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    step=0.1,
                    format="%.1f",
                    help=f"大模型推荐: {llm_help}\n说明: {llm_note_text}"
                )
                custom_max_annual[feat] = new_val
                # 实时更新session_state中的用户自定义值
                st.session_state.user_adjusted_constraints[feat] = new_val

        st.markdown("---")

    # 显示基准信息
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #f0f4ff 0%, #e8f0ff 100%);
                border-radius: 16px; padding: 20px; margin: 15px 0;
                border-left: 5px solid {color_sel};
                box-shadow: 0 4px 15px rgba(0,0,0,0.08);">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <h3 style="margin: 0; color: #2d3748; font-size: 1.3rem;">📌 <b>{country_sel}</b></h3>
                <div style="display: flex; gap: 20px; margin-top: 10px; color: #5a6c7d; flex-wrap: wrap;">
                    <span style="background: {color_sel}22; padding: 4px 12px; border-radius: 8px; font-weight: 500;">{ctype_sel}</span>
                    <span>数据年份: <b>{year_sel}</b></span>
                </div>
            </div>
            <div style="text-align: right; margin-top: 10px;">
                <div style="font-size: 0.85rem; color: #718096;">真实 HALE</div>
                <div style="font-size: 1.8rem; font-weight: 700; color: #2d3748;">{hale0_sel:.2f} <span style="font-size: 0.9rem;">岁</span></div>
            </div>
        </div>
        <div style="margin-top: 15px; padding: 12px; background: white; border-radius: 10px;">
            <span style="color: #718096;">模型预测基准: </span>
            <span style="font-size: 1.3rem; font-weight: 600; color: {color_sel};">{hale_pred_sel:.2f} 岁</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="margin: 25px 0;">', unsafe_allow_html=True)
    target_input = st.slider(
        "🎯 目标 HALE（岁）",
        min_value=max(float(hale_pred_sel + 0.5), slider_min),
        max_value=slider_max,
        value=float(default_target),
        step=0.1,
        format="%.1f",
        help=f"当前模型预测基准为 {hale_pred_sel:.2f} 岁；该国在指标约束下可达上限约 {hale_max_sel:.1f} 岁"
    )

    # 显示该国实际可达上限
    if target_input > hale_max_sel - 0.2:
        st.warning(f"目标 HALE ({target_input:.1f}岁) 已接近或超过该国在指标改善约束下的可达上限 (~{hale_max_sel:.1f}岁)。请适当降低目标。", icon="⚠️")
    gap_preview = target_input - hale_pred_sel

    # 目标预览卡片
    st.markdown(f"""
    <div style="display: flex; gap: 20px; margin: 20px 0; flex-wrap: wrap;">
        <div class="metric-card" style="flex: 1; min-width: 150px; text-align: center; border-top: 4px solid #2193b0;">
            <div style="font-size: 0.9rem; color: #718096;">当前基准</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #2d3748;">{hale_pred_sel:.1f}</div>
            <div style="font-size: 0.85rem; color: #a0aec0;">岁</div>
        </div>
        <div style="display: flex; align-items: center; font-size: 2rem; color: #c0c0c0;">→</div>
        <div class="metric-card" style="flex: 1; min-width: 150px; text-align: center; border-top: 4px solid #27ae60;">
            <div style="font-size: 0.9rem; color: #718096;">目标 HALE</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #27ae60;">{target_input:.1f}</div>
            <div style="font-size: 0.85rem; color: #a0aec0;">岁</div>
        </div>
        <div style="display: flex; align-items: center; font-size: 2rem; color: #c0c0c0;">=</div>
        <div class="metric-card" style="flex: 1; min-width: 150px; text-align: center; border-top: 4px solid {color_sel};">
            <div style="font-size: 0.9rem; color: #718096;">预期提升</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: {color_sel};">+{gap_preview:.1f}</div>
            <div style="font-size: 0.85rem; color: #a0aec0;">岁</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    run = st.button("🚀 开始优化", type="primary", use_container_width=True)

    if run:
        progress_bar = st.progress(0, text="初始化...")
        status_text = st.empty()

        def progress_fn(i, total, strategy, final_h, cost, success):
            pct = int(i / total * 100)
            progress_bar.progress(pct)
            status_text.text(f"  restart {i}/{total}（{strategy}）: HALE={final_h:.3f}y  cost={cost:.2f}  [{'OK' if success else 'no_conv'}]")

        with st.spinner(f"正在为 {country_sel} 运行优化算法（约需 30-60 秒）..."):
            try:
                res = optimize_policy_cli(
                    country_name=country_sel,
                    T=T_input,
                    Hale_input=target_input,
                    year_start=2019,
                    df_panel_data=PANEL,
                    df_cluster_data=CLUSTER,
                    custom_max_annual=custom_max_annual,
                    progress_callback=progress_fn,
                    group_cost_weights=GROUP_COST_WEIGHTS,
                    country_natural_growth=COUNTRY_NATURAL_GROWTH,
                    custom_natural_growth=custom_natural_growth
                )
                progress_bar.empty()
                status_text.empty()

                # 结果展示
                st.markdown("""
                <hr class="custom-divider">
                <h3 class="section-header">✨ 优化结果</h3>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="success-box" style="margin: 15px 0;">
                    <div style="font-size: 1.1rem; font-weight: 600;">✅ 优化完成！</div>
                    <div style="margin-top: 8px; color: #155724;">
                        国家：<b>{country_sel}</b> | 聚类：<b>{res['cluster_type']}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                res1, res2, res3, res4 = st.columns(4)
                with res1:
                    st.markdown(f"""
                    <div class="result-card" style="border-top: 4px solid #3498db;">
                        <div style="font-size: 0.9rem; color: #718096;">模型基准 HALE</div>
                        <div style="font-size: 1.6rem; font-weight: 700; color: #2d3748;">{res['hale_pred_initial']:.2f}</div>
                        <div style="font-size: 0.85rem; color: #a0aec0;">岁</div>
                    </div>
                    """, unsafe_allow_html=True)
                with res2:
                    st.markdown(f"""
                    <div class="result-card" style="border-top: 4px solid #27ae60;">
                        <div style="font-size: 0.9rem; color: #718096;">目标 HALE</div>
                        <div style="font-size: 1.6rem; font-weight: 700; color: #27ae60;">{res['hale_target_input']:.1f}</div>
                        <div style="font-size: 0.85rem; color: #a0aec0;">岁</div>
                    </div>
                    """, unsafe_allow_html=True)
                with res3:
                    gain = res['hale_final'] - res['hale_pred_initial']
                    gain_color = "#27ae60" if gain >= 0 else "#e74c3c"
                    st.markdown(f"""
                    <div class="result-card" style="border-top: 4px solid {gain_color};">
                        <div style="font-size: 0.9rem; color: #718096;">优化后 HALE</div>
                        <div style="font-size: 1.6rem; font-weight: 700; color: {gain_color};">{res['hale_final']:.2f}</div>
                        <div style="font-size: 0.85rem; color: {gain_color};">+{gain:.2f} 岁</div>
                    </div>
                    """, unsafe_allow_html=True)
                with res4:
                    st.markdown(f"""
                    <div class="result-card" style="border-top: 4px solid #9b59b6;">
                        <div style="font-size: 0.9rem; color: #718096;">总成本</div>
                        <div style="font-size: 1.6rem; font-weight: 700; color: #2d3748;">{res['total_cost']:.4f}</div>
                        <div style="font-size: 0.85rem; color: #a0aec0;">单位成本</div>
                    </div>
                    """, unsafe_allow_html=True)

                # 收敛状态
                if res['success']:
                    st.markdown("""
                    <div style="display: inline-block; background: #d4edda; color: #155724;
                                padding: 8px 16px; border-radius: 20px; font-weight: 500; margin-top: 10px;">
                        ✓ 已收敛 - 优化算法成功找到最优解
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="display: inline-block; background: #fff3cd; color: #856404;
                                padding: 8px 16px; border-radius: 20px; font-weight: 500; margin-top: 10px;">
                        △ 未严格收敛 - 结果仍可参考
                    </div>
                    """, unsafe_allow_html=True)

                save_summary_entry(
                    country=country_sel,
                    cluster=res['cluster_type'],
                    initial_hale=res['hale_pred_initial'],
                    target_hale=res['hale_target_input'],  # 使用用户原始输入的目标
                    achieved_hale=res['hale_final'],
                    hale_gain=res['hale_final'] - res['hale_pred_initial'],
                    total_cost=res['total_cost'],
                    converged=res['success']
                )

                # 生成图
                st.markdown("""
                <hr class="custom-divider">
                <h3 class="section-header">📊 最优政策路径</h3>
                """, unsafe_allow_html=True)

                # 获取自然增长率信息
                nat_growth = res.get("natural_growth", {})
                gdp_gr = nat_growth.get("gdp_growth_rate", 0)
                elderly_ch = nat_growth.get("elderly_change_rate", 0)

                fig = plt.figure(figsize=(16, 12))
                fig.patch.set_facecolor('white')
                fig.suptitle(
                    f"Optimal Health Investment Policy — {country_sel} ({res['cluster_type']})\n"
                    f"Target: HALE ≥ {res['hale_target_input']:.1f}y in {T_input} years  | "
                    f"Initial: {res['hale_pred_initial']:.1f}y  | "
                    f"Achieved: {res['hale_final']:.2f}y\n"
                    f"Background Trends: GDP Growth = {gdp_gr*100:.2f}%/yr, Elderly Change = {elderly_ch:+.2f} pts/yr",
                    fontsize=11, fontweight="bold", y=0.98
                )
                gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
                color = CLUSTER_PALETTE.get(res["cluster_type"], "gray")
                years = res["years"]

                ax1 = fig.add_subplot(gs[0, 0])
                ax1.set_facecolor('#f8f9fa')
                ax1.plot(years, res["hales_base"], "k--", lw=2, alpha=0.6, label="Baseline")
                ax1.plot(years, res["hales_opt"],  "-o", color=color, lw=2.5, ms=6, label="Optimal policy")
                ax1.axhline(res["hale_target_input"], color="red", ls=":", lw=2, label=f"Target {res['hale_target_input']:.1f}y")
                ax1.fill_between(years, res["hales_base"], res["hales_opt"], alpha=0.2, color=color)
                ax1.set_title("HALE Trajectory", fontsize=11, fontweight='bold', pad=10)
                ax1.set_xlabel("Year", fontsize=10)
                ax1.set_ylabel("Predicted HALE (years)", fontsize=10)
                ax1.legend(fontsize=9, loc='lower right')
                ax1.grid(alpha=0.25)
                ax1.spines[["top", "right"]].set_visible(False)

                delta_opt = res["delta_opt"]
                delta_df  = pd.DataFrame(
                    delta_opt,
                    columns=[FEATURE_LABELS[f] for f in FEATURE_COLS],
                    index=[f"Y{t+1}" for t in range(T_input)]
                )
                active_cols2 = delta_df.columns[(delta_df.abs() > 1e-4).any()]
                if len(active_cols2) == 0:
                    active_cols2 = delta_df.columns[:3]
                vmax = max(delta_df[active_cols2].abs().values.max(), 1e-6)

                ax2 = fig.add_subplot(gs[0, 1])
                ax2.set_facecolor('#f8f9fa')
                sns.heatmap(delta_df[active_cols2].T, annot=True, fmt=".3f",
                            cmap="RdYlGn", center=0, vmin=-vmax, vmax=vmax,
                            ax=ax2, linewidths=0.5, cbar_kws={"label": "Annual Δ"})
                ax2.set_title("Annual Adjustment Heatmap\n(green=increase, red=decrease)", fontsize=10, fontweight='bold', pad=10)
                ax2.set_xlabel("Planning Year", fontsize=10)
                ax2.tick_params(axis="y", labelsize=9)

                ax3 = fig.add_subplot(gs[1, 0])
                ax3.set_facecolor('#f8f9fa')
                cumulative = delta_opt.sum(axis=0)
                feat_names = [FEATURE_LABELS[f] for f in FEATURE_COLS]
                bar_colors = ["#e74c3c" if v < 0 else color for v in cumulative]
                bars3 = ax3.barh(feat_names, cumulative, color=bar_colors, alpha=0.85, edgecolor='white', height=0.6)
                ax3.axvline(0, color="black", lw=1)
                ax3.set_title(f"Cumulative Adjustment over {T_input} Years", fontsize=10, fontweight='bold', pad=10)
                ax3.set_xlabel("Total Adjustment", fontsize=10)
                ax3.grid(axis="x", alpha=0.25)
                max_abs2 = max(abs(cumulative).max(), 1e-6)
                for j, (val, bar) in enumerate(zip(cumulative, bars3)):
                    if abs(val) > 1e-3:
                        ax3.text(val + np.sign(val) * max_abs2 * 0.02, j,
                                 f"{val:.3f}", va="center", fontsize=9, fontweight='bold')
                ax3.spines[["top", "right"]].set_visible(False)

                ax4 = fig.add_subplot(gs[1, 1])
                ax4.set_facecolor('#f8f9fa')
                costs = res["cost_by_feature"]
                thresh = costs.sum() * 0.005
                mask   = costs > thresh
                if mask.sum() > 0:
                    c_vals3 = list(costs[mask])
                    c_lbls3 = [FEATURE_LABELS[f] for f, m in zip(FEATURE_COLS, mask) if m]
                    if (~mask).any() and costs[~mask].sum() > 0:
                        c_vals3.append(costs[~mask].sum())
                        c_lbls3.append("Others")
                    wedges, texts, autotexts = ax4.pie(c_vals3, labels=c_lbls3, autopct="%1.1f%%",
                            startangle=90, colors=sns.color_palette("Set3", len(c_vals3)))
                    for autotext in autotexts:
                        autotext.set_fontsize(10)
                        autotext.set_fontweight('bold')
                else:
                    ax4.text(0.5, 0.5, "No significant cost", ha="center", va="center", transform=ax4.transAxes, fontsize=12)
                ax4.set_title(f"Cost Breakdown\n(Total = {res['total_cost']:.4f})", fontsize=10, fontweight='bold', pad=10)

                st.pyplot(fig)

                # 政策建议文字
                st.markdown("""
                <hr class="custom-divider">
                <h3 class="section-header">💡 政策建议</h3>
                """, unsafe_allow_html=True)

                st.markdown('<div class="policy-box">', unsafe_allow_html=True)
                top_actions = []
                for i, feat in enumerate(FEATURE_COLS):
                    total_delta = delta_opt[:, i].sum()
                    if abs(total_delta) > 1e-3:
                        if feat == "pm25_exposure":
                            top_actions.append(f"🌬️ **降低 PM2.5 暴露**：年均减少 {abs(total_delta):.2f} μg/m³")
                        elif feat == "safe_water":
                            top_actions.append(f"💧 **提升安全饮水覆盖率**：年均增加 {total_delta:.2f} 个百分点")
                        elif feat == "basic_sanitation":
                            top_actions.append(f"🚽 **改善基础卫生设施**：年均增加 {total_delta:.2f} 个百分点")
                        elif feat == "physicians_per_1000":
                            top_actions.append(f"👨‍⚕️ **增加医生数量**：年均增加 {total_delta:.3f} 名/千人")
                        elif feat == "beds_per_1000":
                            top_actions.append(f"🛏️ **增加医院床位**：年均增加 {total_delta:.3f} 张/千人")
                        elif feat == "health_exp_gdp":
                            top_actions.append(f"💰 **增加卫生支出**：年均增加 GDP 的 {total_delta:.2f}%")
                        elif feat == "G_norm":
                            top_actions.append(f"🏛️ **提升治理水平**：年均提升 {total_delta:.4f}")

                if top_actions:
                    for action in top_actions[:6]:
                        st.markdown(f"<div style='padding: 10px 15px; background: white; border-radius: 10px; margin: 8px 0; border-left: 3px solid #2193b0;'>{action}</div>", unsafe_allow_html=True)
                else:
                    st.info("当前参数配置下无需显著干预即可达到目标。")
                st.markdown('</div>', unsafe_allow_html=True)

                # 保存结果
                os.makedirs(FIG_OPT_DIR, exist_ok=True)
                save_path = os.path.join(FIG_OPT_DIR, f"opt_{country_sel.replace(' ','_')}.png")
                fig.savefig(save_path, dpi=200, bbox_inches="tight")
                st.success(f"图片已保存至：{save_path}")

            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"优化失败：{e}")
                import traceback
                st.code(traceback.format_exc())


# ─────────────────────────────────────────────
# 页面 4：结果汇总
# ─────────────────────────────────────────────
elif page == "📊 结果汇总":
    st.markdown("""
    <div style="text-align: center; padding: 30px 0;">
        <h1 class="main-title">📊 优化结果汇总</h1>
        <p class="sub-title">查看历史优化结果，了解不同国家的健康提升路径与成本效益</p>
    </div>
    """, unsafe_allow_html=True)

    if SUMMARY.empty:
        st.markdown("""
        <div class="info-box" style="margin: 30px auto; max-width: 600px; text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 15px;">📋</div>
            <div style="font-size: 1.1rem; font-weight: 600; color: #2193b0;">暂无历史优化结果</div>
            <div style="color: #718096; margin-top: 10px;">请前往「🎯 健康优化」页面运行优化生成结果</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <hr class="custom-divider">
        <h3 class="section-header">📋 优化结果表</h3>
        """, unsafe_allow_html=True)

        display_df = SUMMARY.copy()
        display_df.insert(0, "#", range(1, len(display_df) + 1))

        # 格式化
        for col in ["Initial_HALE", "Target_HALE", "Achieved_HALE", "HALE_Gain"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")
        if "Total_Cost" in display_df.columns:
            display_df["Total_Cost"] = display_df["Total_Cost"].apply(lambda x: f"{x:.4f}")
        if "Converged" in display_df.columns:
            display_df["Converged"] = display_df["Converged"].map({True: "✓", False: "△"})

        st.markdown('<div class="data-table">', unsafe_allow_html=True)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # 图：初始 vs 达成
        st.markdown("""
        <hr class="custom-divider">
        <h3 class="section-header">📈 优化效果对比</h3>
        """, unsafe_allow_html=True)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.patch.set_facecolor('white')
        countries_s = SUMMARY["Country"].tolist()
        hale_init_s = SUMMARY["Initial_HALE"].tolist()
        hale_ach_s  = SUMMARY["Achieved_HALE"].tolist()
        Hale_tgt    = SUMMARY["Target_HALE"].tolist()
        costs_s     = SUMMARY["Total_Cost"].tolist() if "Total_Cost" in SUMMARY.columns else [0] * len(SUMMARY)
        clusters_s  = SUMMARY["Cluster"].tolist() if "Cluster" in SUMMARY.columns else ["Unknown"] * len(SUMMARY)
        colors_s    = [CLUSTER_PALETTE.get(c, "#888") for c in clusters_s]

        x = np.arange(len(countries_s))
        w = 0.35

        # 子图1：初始 vs 达成
        axes[0].set_facecolor('#f8f9fa')
        axes[0].bar(x - w/2, hale_init_s, w, color="#c0c0c0", edgecolor="white", label="Initial")
        axes[0].bar(x + w/2, hale_ach_s,  w, color=colors_s, edgecolor="white", alpha=0.85, label="Achieved")
        axes[0].plot(x, Hale_tgt, "r*", ms=12, label="Target", zorder=5)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(countries_s, rotation=35, ha="right", fontsize=9)
        axes[0].set_ylabel("HALE (years)", fontsize=10)
        axes[0].set_title("初始 vs 达成 HALE", fontsize=11, fontweight='bold', pad=10)
        axes[0].legend(fontsize=9)
        axes[0].grid(axis="y", alpha=0.3)
        axes[0].spines[["top", "right"]].set_visible(False)

        # 子图2：提升幅度
        gains = [a - i for a, i in zip(hale_ach_s, hale_init_s)]
        axes[1].set_facecolor('#f8f9fa')
        bars2 = axes[1].bar(countries_s, gains, color=colors_s, edgecolor="white", alpha=0.85)
        axes[1].set_xticklabels(countries_s, rotation=35, ha="right", fontsize=9)
        axes[1].set_ylabel("HALE Improvement", fontsize=10)
        axes[1].set_title("HALE 提升幅度", fontsize=11, fontweight='bold', pad=10)
        axes[1].axhline(0, color="black", lw=0.8)
        axes[1].grid(axis="y", alpha=0.3)
        axes[1].spines[["top", "right"]].set_visible(False)
        for bar, gain in zip(bars2, gains):
            if gain > 0:
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                            f'+{gain:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        # 子图3：成本
        axes[2].set_facecolor('#f8f9fa')
        axes[2].bar(countries_s, costs_s, color=colors_s, edgecolor="white", alpha=0.85)
        axes[2].set_xticklabels(countries_s, rotation=35, ha="right", fontsize=9)
        axes[2].set_ylabel("Total Discounted Cost", fontsize=10)
        axes[2].set_title("各国优化成本", fontsize=11, fontweight='bold', pad=10)
        axes[2].grid(axis="y", alpha=0.3)
        axes[2].spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)

        # 查看已生成的图
        st.markdown("""
        <hr class="custom-divider">
        <h3 class="section-header">🖼 已生成优化图</h3>
        """, unsafe_allow_html=True)

        available_imgs = []
        for c in countries_s:
            img_path = os.path.join(FIG_OPT_DIR, f"opt_{c.replace(' ','_')}.png")
            if os.path.exists(img_path):
                available_imgs.append((c, img_path))

        if available_imgs:
            for country_img, img_path in available_imgs:
                st.markdown(f"""
                <div style="background: white; border-radius: 16px; padding: 20px; margin: 15px 0; 
                            box-shadow: 0 4px 15px rgba(0,0,0,0.08);">
                    <h4 style="color: #2d3748; margin-bottom: 15px;">📌 {country_img}</h4>
                """, unsafe_allow_html=True)
                from PIL import Image
                st.image(Image.open(img_path), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box" style="margin: 20px 0;">
                <div style="font-size: 1rem;">暂无生成的优化图，请先在「🎯 健康优化」页面运行。</div>
            </div>
            """, unsafe_allow_html=True)
