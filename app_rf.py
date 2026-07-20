import joblib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

# ==========================================
# 1. 变量映射与类型配置
# ==========================================
# 二分类变量定义
BINARY_VARS = ["sepsis", "AKI", "CRRT", "ventilation", "antibiotic"]
# 连续变量分类（用于区分整数和小数）
INT_VARS = ["HR", "RR", "SOFA", "SAPS_II"]  # 临床上表现为整数
FLOAT_VARS = ["ALB"]  # 临床上可能有小数

# ==========================================
# 2. 页面与紧凑样式配置
# ==========================================
st.set_page_config(page_title="ARDS Risk Prediction Tool", layout="wide")

# 自定义 CSS 压缩全局上下间距
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    </style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# 3. 加载模型与配置
# ==========================================
@st.cache_resource
def load_assets():
    model = joblib.load("model_deploy.pkl")
    features = joblib.load("feature_names.pkl")
    defaults = joblib.load("defaults.pkl")
    return model, features, defaults


model, feature_names, defaults = load_assets()

# ==========================================
# 4. 界面逻辑
# ==========================================
st.markdown(
    "<h2 style='text-align: center; margin-bottom: 1rem;'>ARDS Risk Prediction"
    " System</h2>",
    unsafe_allow_html=True,
)

user_input_values = {}

# 左右两列分布，左侧参数输入框稍宽一些，更美观
col_input, col_result = st.columns([1.1, 1], gap="large")

# ----------------- 左侧：输入区域 -----------------
with col_input:
    st.markdown("#### 🏥 Clinical Parameters")
    with st.container(border=True):
        # 1. 处理数值变量（一行两列，紧凑排版）
        st.caption("Continuous Variables & Scores")
        num_cols = st.columns(2)
        all_numerics = INT_VARS + FLOAT_VARS

        for i, var in enumerate(all_numerics):
            col_idx = i % 2
            with num_cols[col_idx]:
                if var in INT_VARS:
                    # 心率、呼吸频率、评分设置为整数
                    user_input_values[var] = st.number_input(
                        f"{var} (Default: {int(defaults[var])})",
                        value=int(defaults[var]),
                        step=1,
                        format="%d",
                    )
                else:
                    # 白蛋白设置为浮点数
                    user_input_values[var] = st.number_input(
                        f"{var} (Default: {float(defaults[var]):.2f})",
                        value=float(defaults[var]),
                        step=0.1,
                        format="%.2f",
                    )

        st.divider()  # 分割线

        # 2. 处理分类变量（一行两列，紧凑排版）
        st.caption("Categorical Variables")
        bin_cols = st.columns(2)
        for i, var in enumerate(BINARY_VARS):
            col_idx = i % 2
            with bin_cols[col_idx]:
                val = st.radio(
                    f"{var}",
                    options=["No (0)", "Yes (1)"],
                    horizontal=True,
                    key=f"radio_{var}",
                )
                user_input_values[var] = 1 if val == "Yes (1)" else 0

# ----------------- 右侧：预测与解释区域 -----------------
with col_result:
    st.markdown("#### 📊 Prediction Result")

    # 【核心修改点】让按钮横跨整个右侧列，不再与文字同行挤压
    calculate_btn = st.button(
        "🚀 Calculate ARDS Risk", type="primary", use_container_width=True
    )

    if calculate_btn:
        input_df = pd.DataFrame([user_input_values], columns=feature_names)

        # 预测
        pred_prob = model.predict_proba(input_df)[0][1]
        risk = pred_prob * 100

        # 独立显示结果，与上方按钮和下方仪表盘保持清晰间距
        st.metric(label="Calculated ARDS Probability", value=f"{risk:.1f}%")

        # 紧凑型仪表盘 (压缩高度和边缘空白)
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=risk,
                number={"suffix": "%", "font": {"size": 36}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {
                        "color": "#d62728" if risk > 30 else "#2ca02c",
                        "thickness": 0.3,
                    },
                    "bgcolor": "white",
                    "borderwidth": 1,
                    "bordercolor": "gray",
                },
            )
        )
        fig.update_layout(height=180, margin=dict(l=20, r=20, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        # SHAP 解释 (高度压缩，消除白边)
        st.markdown(
            "**Model Interpretation (SHAP Value)**",
            help="Red arrows push risk higher; Blue arrows push risk lower.",
        )

        base_model = (
            model.calibrated_classifiers_[0].estimator
            if hasattr(model, "calibrated_classifiers_")
            else model
        )
        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer(input_df)

        fig_shap, ax = plt.subplots(figsize=(6, 2.2))
        shap.plots.waterfall(shap_values[0, :, 1], max_display=6, show=False)
        plt.tight_layout()  # 自动裁切多余留白
        st.pyplot(fig_shap, use_container_width=True)
    else:
        st.info("👈 Please adjust parameters on the left and click calculate.")