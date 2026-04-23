import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# =========================================================
# 0. 基本设置
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
EXCEL_FILE = BASE_DIR / "full data.xlsx"
OUT_DIR = BASE_DIR / "peter_style_pfr_fit"
OUT_DIR.mkdir(exist_ok=True)

R = 8.314462618
EPS = 1e-12

# 论文里用的是平均温度参数化，这里用你数据的中间温度
T_REF = 513.15

# 是否只拟合部分数据
USE_GHSV_FILTER = False
GHSV_FILTER = [4000, 8000, 12000]

USE_T_FILTER = False
T_MIN = 493.15
T_MAX = 533.15

# 是否在甲醇反应里保留 (1-beta1)
# 论文里 power law 用了平衡项，但也指出甲醇那一路通常远离平衡
USE_BETA1 = True

# 残差权重
W_ME0H = 1.0
W_H2O = 1.0

# 可选：对 outlet fugacity 加一点辅助约束，避免解太漂
USE_FUGACITY_AUX = True
W_FUGACITY_AUX = 0.15


# =========================================================
# 1. 读取数据
# =========================================================
def load_data():
    """
    读取你的 Excel。
    假设第二行是单位，所以 skiprows=[1]
    """
    df = pd.read_excel(EXCEL_FILE, skiprows=[1])
    df.columns = df.columns.str.strip()

    df = df.rename(columns={
        "r CH3OH": "rMeOH",
        "r CO": "rCO",
        "r CO2": "rCO2"
    })

    required_cols = [
        "H/C", "p", "GHSV", "T",
        "fCO2", "fH2", "fCH3OH", "fH2O", "fCO",
        "rMeOH", "rCO"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"缺少列: {col}")

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required_cols).copy()

    if USE_GHSV_FILTER:
        df = df[df["GHSV"].isin(GHSV_FILTER)].copy()

    if USE_T_FILTER:
        df = df[(df["T"] >= T_MIN) & (df["T"] <= T_MAX)].copy()

    # 按论文思路拟合甲醇和水生成速率
    # 对于 CO2+H2 -> MeOH+H2O 以及 CO2+H2 -> CO+H2O，
    # 有：r_H2O = r_MeOH + r_CO
    df["rH2O_meas"] = df["rMeOH"] + df["rCO"]

    df = df.reset_index(drop=True)

    print("Excel 路径:", EXCEL_FILE)
    print("总数据点数:", len(df))
    print("GHSV 分组:", sorted(df["GHSV"].unique()))
    print("温度点:", sorted(df["T"].unique()))
    return df


# =========================================================
# 2. 平衡常数
# =========================================================
def calc_keq(T):
    """
    沿用你之前一直在用的平衡常数表达式。
    这样和你现有 fugacity 单位体系更一致。
    """
    K1 = np.exp(
        1.6654 + 4553.34 / T - 2.72613 * np.log(T)
        - 1.422914e-2 * T + 0.172060e-4 * T**2
        - 1.106294e-8 * T**3 + 0.319698e-11 * T**4
    ) * (0.101325) ** (-2)

    K2 = np.exp(
        -11.4998 - 4649.92 / T + 3.2066 * np.log(T)
        - 0.0107251 * T + 0.697955e-5 * T**2
        - 0.336848e-8 * T**3 + 0.811184e-12 * T**4
    )
    return K1, K2


# =========================================================
# 3. 入口流量重建
# =========================================================
def compute_inlet_flows(row, F_total=1.0):
    """
    先做最小版本：
    假设入口只有 CO2 + H2

    H/C = H2 / CO2

    所以：
        y_CO2 = 1 / (1 + H/C)
        y_H2  = H/C / (1 + H/C)

    返回 5 个组分流量：
        [CO2, H2, CH3OH, H2O, CO]
    """
    hc = float(row["H/C"])

    y_co2 = 1.0 / (1.0 + hc)
    y_h2 = hc / (1.0 + hc)

    F0 = np.array([
        y_co2 * F_total,
        y_h2 * F_total,
        0.0,
        0.0,
        0.0
    ], dtype=float)

    return F0


# =========================================================
# 4. fugacity 计算
# =========================================================
def fugacity_from_flows(F, T, P_MPa):
    """
    当前先用理想近似：
        f_i ≈ y_i * P

    这里保留接口，后面你可以替换成 SRK/PR/EOS。
    """
    F = np.maximum(F, EPS)
    Ft = np.sum(F)
    y = F / Ft
    f = y * P_MPa
    return f


# =========================================================
# 5. Peter 风格的局部 power law
# =========================================================
def unpack_params(params):
    """
    参数说明：

    lnA1, Ea1, a_CO2, a_H2, a_H2O,
    lnA2, Ea2, c_CO2, c_H2,
    ln_tau_scale

    其中：
    - 反应 1: MeOH
    - 反应 2: RWGS
    - ln_tau_scale: 用来把 1/GHSV 映射成有效停留尺度
      这是为了适配你的数据新增的参数，不是论文原文参数
    """
    return params


def local_rates(fCO2, fH2, fCH3OH, fH2O, fCO, T, params):
    """
    按 Peter 论文的 power-law 思路做“适配版”：

    论文原始 power law 包括：
    - H2, CO2 驱动
    - 水抑制
    - 平衡项
    - 某些产品项最终被发现可以省略

    对你的数据，这里采用最稳的改写：

    r1 = k1 * fCO2^a * fH2^b / (n_w + fH2O)^w * (1-beta1) [可选]
    r2 = k2 * fCO2^c * fH2^d * (1-beta2)

    这样既保留论文里“水抑制 + 平衡驱动”的思想，
    又避免把太多产品指数放进分子里导致冗余。
    """
    lnA1, Ea1, a_co2, a_h2, a_h2o, lnA2, Ea2, c_co2, c_h2, ln_tau_scale = unpack_params(params)

    # 速率常数，采用 Tav/Tref 重参数化
    k1 = np.exp(lnA1) * np.exp(-Ea1 / R * (1.0 / T - 1.0 / T_REF))
    k2 = np.exp(lnA2) * np.exp(-Ea2 / R * (1.0 / T - 1.0 / T_REF))

    K1, K2 = calc_keq(T)

    fCO2 = np.maximum(fCO2, EPS)
    fH2 = np.maximum(fH2, EPS)
    fCH3OH = np.maximum(fCH3OH, EPS)
    fH2O = np.maximum(fH2O, EPS)
    fCO = np.maximum(fCO, EPS)

    # approach to equilibrium
    beta1 = (fCH3OH * fH2O) / (K1 * fCO2 * fH2**3 + 1e-30)
    beta2 = (fCO * fH2O) / (K2 * fCO2 * fH2 + 1e-30)

    beta1 = np.clip(beta1, 0.0, 0.999999)
    beta2 = np.clip(beta2, 0.0, 0.999999)

    # 水抑制项，模仿 Peter 文中 n + fH2O 的处理
    n_w = 1e-3

    # MeOH 路
    r1 = k1 * (fCO2 ** a_co2) * (fH2 ** a_h2) / ((n_w + fH2O) ** a_h2o)
    if USE_BETA1:
        r1 = r1 * (1.0 - beta1)

    # RWGS 路
    r2 = k2 * (fCO2 ** c_co2) * (fH2 ** c_h2) * (1.0 - beta2)

    r1 = max(r1, 0.0)
    r2 = max(r2, 0.0)

    return r1, r2


# =========================================================
# 6. PFR ODE
# =========================================================
def pfr_odes(z, F, T, P_MPa, params):
    """
    z 取 0->1 的无量纲床层坐标
    真正的停留尺度通过 tau_scale / GHSV 体现在右端项前面
    """
    F = np.maximum(F, EPS)

    fCO2, fH2, fCH3OH, fH2O, fCO = fugacity_from_flows(F, T, P_MPa)
    r1, r2 = local_rates(fCO2, fH2, fCH3OH, fH2O, fCO, T, params)

    lnA1, Ea1, a_co2, a_h2, a_h2o, lnA2, Ea2, c_co2, c_h2, ln_tau_scale = unpack_params(params)
    tau_scale = np.exp(ln_tau_scale)

    # 用 tau_scale / GHSV 表示有效停留尺度
    # 这里 z 是 0->1，所以真正的“强度”乘在右端项前面
    # 注意：GHSV 在每个实验点不同，因此在外层传入时处理
    # 这里先只返回“局部反应贡献”，外层再乘 scale
    dFdz_local = np.array([
        -r1 - r2,         # CO2
        -3.0 * r1 - r2,   # H2
        +r1,              # CH3OH
        +r1 + r2,         # H2O
        +r2               # CO
    ], dtype=float)

    return dFdz_local


def simulate_one_experiment(row, params):
    """
    对单个实验点做 PFR 积分。
    关键做法：
    - z 用 0 -> 1
    - 真正尺度 = tau_scale / GHSV
    """
    T = float(row["T"])
    P = float(row["p"])
    ghsv = float(row["GHSV"])

    lnA1, Ea1, a_co2, a_h2, a_h2o, lnA2, Ea2, c_co2, c_h2, ln_tau_scale = unpack_params(params)
    tau_scale = np.exp(ln_tau_scale)

    # 有效尺度
    scale = tau_scale / ghsv

    F0 = compute_inlet_flows(row, F_total=1.0)

    def rhs(z, F):
        return scale * pfr_odes(z, F, T, P, params)

    sol = solve_ivp(
        fun=rhs,
        t_span=(0.0, 1.0),
        y0=F0,
        method="BDF",
        rtol=1e-7,
        atol=1e-10
    )

    if (not sol.success) or np.any(~np.isfinite(sol.y)):
        raise RuntimeError("ODE integration failed")

    F_out = np.maximum(sol.y[:, -1], EPS)
    f_out = fugacity_from_flows(F_out, T, P)

    # 由 PFR 结果构造“积分速率”
    # 因为你的实验表里本来就是以速率形式给出的
    rMeOH_pred = (F_out[2] - F0[2]) / scale
    rH2O_pred = (F_out[3] - F0[3]) / scale
    rCO_pred = (F_out[4] - F0[4]) / scale

    return F_out, f_out, rMeOH_pred, rH2O_pred, rCO_pred


# =========================================================
# 7. 残差函数：按论文思路拟合积分速率
# =========================================================
def residual_vector(params, df):
    """
    按 Peter 论文的 spirit：
    主要拟合积分甲醇速率 + 水生成速率

    你的数据没有直接给 rH2O，因此：
        rH2O_meas = rMeOH + rCO

    可选地加入一点 outlet fugacity 残差作为辅助约束，
    防止只靠速率时解太漂。
    """
    residuals = []

    for _, row in df.iterrows():
        try:
            F_out, f_out, rMeOH_pred, rH2O_pred, rCO_pred = simulate_one_experiment(row, params)

            rMeOH_meas = float(row["rMeOH"])
            rH2O_meas = float(row["rH2O_meas"])

            res_meoh = W_ME0H * (rMeOH_pred - rMeOH_meas) / max(abs(rMeOH_meas), 1e-6)
            res_h2o = W_H2O * (rH2O_pred - rH2O_meas) / max(abs(rH2O_meas), 1e-6)

            residuals.append(res_meoh)
            residuals.append(res_h2o)

            if USE_FUGACITY_AUX:
                f_meas = np.array([row["fCO2"], row["fH2"], row["fCH3OH"], row["fH2O"], row["fCO"]], dtype=float)
                res_f = W_FUGACITY_AUX * (f_out - f_meas) / np.maximum(np.abs(f_meas), 1e-6)
                residuals.extend(res_f.tolist())

        except Exception:
            # 若某个点失败，给大惩罚
            residuals.extend([100.0, 100.0])
            if USE_FUGACITY_AUX:
                residuals.extend([100.0] * 5)

    return np.array(residuals, dtype=float)


# =========================================================
# 8. 拟合
# =========================================================
def fit_model(df):
    """
    论文里是 weighted nonlinear LS + trust region。
    这里直接用 least_squares(method='trf') 来对应。
    """
    # 初值
    p0 = np.array([
        -8.0,  80000.0,  0.5, 1.5, 0.5,   # MeOH
        -5.0,  60000.0,  0.5, 1.0,  0.0   # RWGS + tau_scale
    ], dtype=float)

    # 下界
    lb = np.array([
        -15.0, 20000.0, 0.0, 0.5, 0.0,
        -15.0, 20000.0, 0.0, -1.0, -10.0
    ], dtype=float)

    # 上界
    ub = np.array([
        5.0, 130000.0, 2.0, 3.5, 2.0,
        5.0, 100000.0, 2.0, 2.0, 10.0
    ], dtype=float)

    result = least_squares(
        fun=residual_vector,
        x0=p0,
        bounds=(lb, ub),
        args=(df,),
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
        max_nfev=600,
        verbose=2
    )

    return result


# =========================================================
# 9. 用拟合参数预测整张表
# =========================================================
def predict_dataset(df, params):
    rows = []

    for _, row in df.iterrows():
        row_out = row.copy()

        try:
            F_out, f_out, rMeOH_pred, rH2O_pred, rCO_pred = simulate_one_experiment(row, params)

            row_out["fCO2_pred"] = f_out[0]
            row_out["fH2_pred"] = f_out[1]
            row_out["fCH3OH_pred"] = f_out[2]
            row_out["fH2O_pred"] = f_out[3]
            row_out["fCO_pred"] = f_out[4]

            row_out["rMeOH_pred"] = rMeOH_pred
            row_out["rH2O_pred"] = rH2O_pred
            row_out["rCO_pred"] = rCO_pred

        except Exception:
            for c in ["fCO2_pred", "fH2_pred", "fCH3OH_pred", "fH2O_pred", "fCO_pred",
                      "rMeOH_pred", "rH2O_pred", "rCO_pred"]:
                row_out[c] = np.nan

        rows.append(row_out)

    return pd.DataFrame(rows)


# =========================================================
# 10. 指标
# =========================================================
def calc_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return np.nan

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot < 1e-12:
        return np.nan
    return 1.0 - ss_res / ss_tot


def calc_rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return np.nan

    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def build_summary_metrics(df_pred):
    rows = []

    for ghsv, part in df_pred.groupby("GHSV"):
        row = {"GHSV": ghsv, "n_points": len(part)}

        for true_col, pred_col in [
            ("rMeOH", "rMeOH_pred"),
            ("rH2O_meas", "rH2O_pred"),
            ("rCO", "rCO_pred"),
            ("fCO2", "fCO2_pred"),
            ("fH2", "fH2_pred"),
            ("fCH3OH", "fCH3OH_pred"),
            ("fH2O", "fH2O_pred"),
            ("fCO", "fCO_pred")
        ]:
            row[f"R2_{true_col}"] = calc_r2(part[true_col], part[pred_col])
            row[f"RMSE_{true_col}"] = calc_rmse(part[true_col], part[pred_col])

        rows.append(row)

    return pd.DataFrame(rows)


# =========================================================
# 11. 画图
# =========================================================
def parity_plot(df_pred, true_col, pred_col, title, file_name):
    plt.figure(figsize=(6, 6))

    for ghsv, part in df_pred.groupby("GHSV"):
        plt.scatter(part[true_col], part[pred_col], label=f"GHSV={int(ghsv)}", alpha=0.75)

    all_true = df_pred[true_col].to_numpy(dtype=float)
    all_pred = df_pred[pred_col].to_numpy(dtype=float)
    mask = np.isfinite(all_true) & np.isfinite(all_pred)

    all_true = all_true[mask]
    all_pred = all_pred[mask]

    mn = min(all_true.min(), all_pred.min())
    mx = max(all_true.max(), all_pred.max())

    plt.plot([mn, mx], [mn, mx], "k--")
    plt.xlabel(f"Measured {true_col}")
    plt.ylabel(f"Predicted {pred_col}")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / file_name, dpi=300)
    plt.show()


# =========================================================
# 12. 主程序
# =========================================================
def main():
    df = load_data()

    result = fit_model(df)

    print("\n==============================")
    print("拟合完成")
    print("==============================")
    print("success =", result.success)
    print("status  =", result.status)
    print("message =", result.message)
    print("cost    =", result.cost)

    names = [
        "lnA1", "Ea1", "a_CO2", "a_H2", "a_H2O",
        "lnA2", "Ea2", "c_CO2", "c_H2", "ln_tau_scale"
    ]

    print("\n拟合参数：")
    for n, v in zip(names, result.x):
        print(f"{n} = {v:.6f}")

    param_df = pd.DataFrame({
        "param_name": names,
        "value": result.x
    })
    param_df.to_excel(OUT_DIR / "fitted_parameters.xlsx", index=False)

    df_pred = predict_dataset(df, result.x)
    df_pred.to_excel(OUT_DIR / "predicted_dataset.xlsx", index=False)

    summary_df = build_summary_metrics(df_pred)
    summary_df.to_excel(OUT_DIR / "grouped_fit_metrics.xlsx", index=False)

    print("\n按 GHSV 分组的指标：")
    print(summary_df)

    # 速率 parity
    parity_plot(df_pred, "rMeOH", "rMeOH_pred", "Parity Plot: rMeOH", "parity_rMeOH.png")
    parity_plot(df_pred, "rH2O_meas", "rH2O_pred", "Parity Plot: rH2O", "parity_rH2O.png")
    parity_plot(df_pred, "rCO", "rCO_pred", "Parity Plot: rCO", "parity_rCO.png")

    # fugacity parity
    parity_plot(df_pred, "fCO2", "fCO2_pred", "Parity Plot: fCO2", "parity_fCO2.png")
    parity_plot(df_pred, "fH2", "fH2_pred", "Parity Plot: fH2", "parity_fH2.png")
    parity_plot(df_pred, "fCH3OH", "fCH3OH_pred", "Parity Plot: fCH3OH", "parity_fCH3OH.png")
    parity_plot(df_pred, "fH2O", "fH2O_pred", "Parity Plot: fH2O", "parity_fH2O.png")
    parity_plot(df_pred, "fCO", "fCO_pred", "Parity Plot: fCO", "parity_fCO.png")

    print("\n输出文件已保存到：")
    print(OUT_DIR)


if __name__ == "__main__":
    main()