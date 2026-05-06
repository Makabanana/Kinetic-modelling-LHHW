# import list
import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #画图
import streamlit as st  #做网页界面的工具

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# 1. Basic set
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
R = 8.314462618  # J/mol/K
RATE_COLS = ["r_CO", "r_CH3OH"]     #定义允许选择的速率列
EXPECTED_COLUMNS = [
    "H/C", "p", "GHSV", "T_C", "T_K",
    "fCO2", "fH2", "fCH3OH", "fH2O", "fCO",
    "r_CH3OH", "r_CO", "r_CO2"
]                                   #定义一个列表,excel读进来的列名

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#2. Streamlit 页面设置
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#为它起名
st.set_page_config(page_title="Arrhenius Fitting Tool", layout="wide")

#美化样式
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1.2rem;
    }
    div[data-testid="stDataFrame"] * {
        font-size: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#3. 从excel读取数字
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#定义一个函数，名字叫 load_data
#输入是上传的文件对象，输出是一个 pandas DataFrame
def load_data(file_obj) -> pd.DataFrame:
    raw = pd.read_excel(file_obj)                       #pd读数据
    df = raw.iloc[1:].copy()                            #从第二行再取数据
    df.columns = EXPECTED_COLUMNS

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["H/C", "p", "GHSV", "T_K"])
    return df                                           #返回dataframe

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#4. 核心拟合逻辑-fit_arrhenius函数
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#定义 Arrhenius 拟合函数
#输入是一组子数据 sub_df，以及你想拟合的速率列名 rate_col
#输出是一个字典，里面装着 Ea、lnA、R² 等结果

def fit_arrhenius(sub_df: pd.DataFrame, rate_col: str) -> dict:
    sub = sub_df.sort_values("T_K").copy()          #先按TK从小到大排序
    sub = sub[sub[rate_col] > 0].copy()             #只保留速率大于 0 的点

    if len(sub) < 3:
        raise ValueError("positive points < 3, cannot fit")  #筛掉负速率点

    x = 1.0 / sub["T_K"].to_numpy()
    y = np.log(sub[rate_col].to_numpy())            #把温度列变成 1/T，并转成 numpy 数组。速率取对数

    slope, intercept = np.polyfit(x, y, 1)      #用一次多项式拟合，也就是直线拟合：
    y_pred = slope * x + intercept

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))          #算误差

    ea_j_mol = -slope * R
    ea_kj_mol = ea_j_mol / 1000.0
    ln_a = intercept
    a = np.exp(ln_a)                                      #换算单位，并把lnA还原到A


#输出字典
    return {
        "n_points": len(sub),
        "T_min_C": sub["T_C"].min(),
        "T_max_C": sub["T_C"].max(),
        "lnA": ln_a,
        "A": a,
        "Ea_J_mol": ea_j_mol,
        "Ea_kJ_mol": ea_kj_mol,
        "R2_ln": r2,
        "RMSE_ln": rmse,
    }


#批量分组拟合
#定义一个批量拟合函数。
#输入总数据表和你选择的速率列。
#输出所有组的拟合汇总表。

def batch_fit_selected(df: pd.DataFrame, selected_rates: list[str]) -> pd.DataFrame:
    records = []            #先建一个空列表，用来收集每个分组的结果。
    grouped = df.groupby(["p", "H/C", "GHSV"], dropna=True)         #按照 p、H/C、GHSV 分组。

    for (p_value, hc_value, ghsv_value), sub in grouped:
        for rate_col in selected_rates:
            try:
                result = fit_arrhenius(sub, rate_col)
                row = {
                    "p": p_value,
                    "H/C": hc_value,
                    "GHSV": ghsv_value,
                    "rate": rate_col,
                    **result,           #把 fit_arrhenius 返回的整个字典内容展开并塞进这一行。字典合并写法
                    "status": "ok",
                }
            except Exception as e:      #如果失败 也要保存
                row = {
                    "p": p_value,
                    "H/C": hc_value,
                    "GHSV": ghsv_value,
                    "rate": rate_col,
                    "n_points": np.nan,
                    "T_min_C": np.nan,
                    "T_max_C": np.nan,
                    "lnA": np.nan,
                    "A": np.nan,
                    "Ea_J_mol": np.nan,
                    "Ea_kJ_mol": np.nan,
                    "R2_ln": np.nan,
                    "RMSE_ln": np.nan,
                    "status": str(e),
                }
            records.append(row)

    return pd.DataFrame(records)            #总的一个列表


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#5. 后处理与画图
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#生成工况标签

def make_condition_key(df: pd.DataFrame) -> pd.Series:
    return (
        "H/C=" + df["H/C"].astype(str)
        + " | p=" + df["p"].astype(str)
        + " | GHSV=" + df["GHSV"].astype(str)
    )


#画 Ea 对 lnA 总图

def fig_ea_lna(summary_ok: pd.DataFrame, annotate: bool, annotation_size: int):
    fig, ax = plt.subplots(figsize=(7.5, 5.6))

    for rate_col, part in summary_ok.groupby("rate"):
        ax.scatter(part["Ea_kJ_mol"], part["lnA"], s=50, label=rate_col)
        if annotate:
            for _, row in part.iterrows():
                label = f"H/C={row['H/C']}, p={row['p']}, GHSV={row['GHSV']}"
                ax.annotate(
                    label,
                    (row["Ea_kJ_mol"], row["lnA"]),
                    fontsize=annotation_size,
                    xytext=(3, 3),
                    textcoords="offset points",
                )

    ax.set_xlabel("Ea (kJ/mol)")
    ax.set_ylabel("lnA")
    ax.set_title("Selected groups: Ea vs lnA")
    ax.legend()
    fig.tight_layout()
    return fig

#画 单组拟合细节图

def fig_arrhenius_detail(raw_df: pd.DataFrame, rate_col: str):
    sub = raw_df.sort_values("T_K").copy()
    sub = sub[sub[rate_col] > 0].copy()

    if len(sub) < 3:
        return None, "positive points < 3, cannot fit"

    x = 1.0 / sub["T_K"].to_numpy()
    y = np.log(sub[rate_col].to_numpy())
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.scatter(x, y, s=45, label="data")
    ax.plot(x, y_pred, linewidth=1.8, label="fit")
    ax.set_xlabel("1 / T (1/K)")
    ax.set_ylabel(f"ln({rate_col})")
    ax.set_title("Arrhenius fit detail")
    ax.legend()
    fig.tight_layout()
    return fig, None


#下载 CSV
def dataframe_download_button(df: pd.DataFrame, filename: str, label: str):
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(label=label, data=csv_bytes, file_name=filename, mime="text/csv")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#6. 网页设计
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#页面标题和说明
st.title("Arrhenius 拟合小工具")
st.caption("上传 Excel，勾选 H/C、p、GHSV 后自动拟合，并查看不同工况下的 Ea、lnA、R² 和 Arrhenius 直线。")

uploaded_file = st.file_uploader("上传你的 Excel 文件", type=["xlsx", "xls"])

with st.expander("文件格式说明", expanded=False):
    st.write(
        "默认沿用你现有脚本的读法：第一行是单位，第二行开始是真实数据；列顺序应为 H/C, p, GHSV, T_C, T_K, fCO2, fH2, fCH3OH, fH2O, fCO, r_CH3OH, r_CO, r_CO2。"
    )

if uploaded_file is None:
    st.info("先上传一个 Excel 文件再开始。")
    st.stop()

try:
    df = load_data(uploaded_file)
except Exception as e:
    st.error(f"文件读取失败：{e}")
    st.stop()

if df.empty:
    st.warning("读到的数据为空，请检查文件内容。")
    st.stop()

st.sidebar.header("筛选条件")

all_hc = sorted(df["H/C"].dropna().unique().tolist())
all_p = sorted(df["p"].dropna().unique().tolist())
all_ghsv = sorted(df["GHSV"].dropna().unique().tolist())

selected_hc = st.sidebar.multiselect("选择 H/C", all_hc, default=all_hc)
selected_p = st.sidebar.multiselect("选择 p", all_p, default=all_p)
selected_ghsv = st.sidebar.multiselect("选择 GHSV", all_ghsv, default=all_ghsv)
selected_rates = st.sidebar.multiselect("选择速率列", RATE_COLS, default=RATE_COLS)
annotate_points = st.sidebar.checkbox("在 Ea-lnA 图上显示工况文字", value=True)
annotation_size = st.sidebar.slider("标注文字大小", min_value=4, max_value=12, value=6, step=1)
show_raw = st.sidebar.checkbox("显示筛选后的原始数据", value=False)

filtered = df[
    df["H/C"].isin(selected_hc)
    & df["p"].isin(selected_p)
    & df["GHSV"].isin(selected_ghsv)
].copy()

if filtered.empty:
    st.warning("当前筛选下没有数据。")
    st.stop()

filtered["condition_key"] = make_condition_key(filtered)

st.subheader("当前筛选概况")
col1, col2, col3 = st.columns(3)
col1.metric("筛选后数据点数", len(filtered))
col2.metric("筛选后工况组数", filtered[["H/C", "p", "GHSV"]].drop_duplicates().shape[0])
col3.metric("参与拟合的速率数", len(selected_rates))

if show_raw:
    st.subheader("筛选后的原始数据")
    st.dataframe(filtered, use_container_width=True, height=260)

summary = batch_fit_selected(filtered, selected_rates)
summary_ok = summary[summary["status"] == "ok"].copy()
summary_bad = summary[summary["status"] != "ok"].copy()

st.subheader("拟合结果总表")
if summary_ok.empty:
    st.warning("没有成功拟合的组。通常是因为某些组的正速率点少于 3 个。")
else:
    display_cols = [
        "rate", "H/C", "p", "GHSV", "n_points", "T_min_C", "T_max_C",
        "Ea_kJ_mol", "lnA", "A", "R2_ln", "RMSE_ln"
    ]
    st.dataframe(
        summary_ok[display_cols].sort_values(["rate", "H/C", "p", "GHSV"]),
        use_container_width=True,
        height=320,
    )
    dataframe_download_button(summary_ok, "arrhenius_summary_ok.csv", "下载成功拟合结果 CSV")

if not summary_bad.empty:
    with st.expander("查看未成功拟合的组"):
        st.dataframe(summary_bad, use_container_width=True, height=200)
        dataframe_download_button(summary_bad, "arrhenius_summary_failed.csv", "下载失败拟合结果 CSV")

if not summary_ok.empty:
    st.subheader("Ea vs lnA 总图")
    fig1 = fig_ea_lna(summary_ok, annotate=annotate_points, annotation_size=annotation_size)
    st.pyplot(fig1)

    buf = io.BytesIO()
    fig1.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    st.download_button(
        label="下载 Ea-lnA 图片 PNG",
        data=buf.getvalue(),
        file_name="Ea_lnA_selected.png",
        mime="image/png",
    )

    st.subheader("单组 Arrhenius 细节图")
    detail_options = summary_ok.copy()
    detail_options["detail_label"] = (
        detail_options["rate"]
        + " | H/C=" + detail_options["H/C"].astype(str)
        + " | p=" + detail_options["p"].astype(str)
        + " | GHSV=" + detail_options["GHSV"].astype(str)
    )

    chosen_label = st.selectbox("选择一个工况查看 ln(r) vs 1/T 拟合直线", detail_options["detail_label"].tolist())
    chosen_row = detail_options.loc[detail_options["detail_label"] == chosen_label].iloc[0]

    raw_sub = filtered[
        (filtered["H/C"] == chosen_row["H/C"])
        & (filtered["p"] == chosen_row["p"])
        & (filtered["GHSV"] == chosen_row["GHSV"])
    ].copy()

    fig2, err = fig_arrhenius_detail(raw_sub, chosen_row["rate"])
    if err:
        st.warning(err)
    else:
        left, right = st.columns([1.45, 1])
        with left:
            st.pyplot(fig2)
        with right:
            st.markdown("**该组拟合参数**")
            st.write(f"rate: {chosen_row['rate']}")
            st.write(f"H/C: {chosen_row['H/C']}")
            st.write(f"p: {chosen_row['p']}")
            st.write(f"GHSV: {chosen_row['GHSV']}")
            st.write(f"Ea: {chosen_row['Ea_kJ_mol']:.4f} kJ/mol")
            st.write(f"lnA: {chosen_row['lnA']:.4f}")
            st.write(f"A: {chosen_row['A']:.4e}")
            st.write(f"R²: {chosen_row['R2_ln']:.4f}")
            st.write(f"RMSE: {chosen_row['RMSE_ln']:.4f}")

st.subheader("怎么运行")
st.code(
    "pip install streamlit pandas numpy matplotlib openpyxl\n"
    "streamlit run arrhenius_streamlit_app.py",
    language="bash",
)

st.caption("如果你下一步想要，我还可以继续给这个小工具加上：固定某一个变量只看另外两个变量的趋势、自动批量导出每组拟合图、R²筛选、或者只显示你勾选的单个 rate。")
