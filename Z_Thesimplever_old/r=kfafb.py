import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

R = 8.314  # J/mol/K


# =========================
# 1. 读数据
# =========================
file_path = "fulldata.xlsx"   # 改成你的文件路径
sheet_name = "full"

raw = pd.read_excel(file_path, sheet_name=sheet_name)

# 去掉列名前后空格
raw.columns = [c.strip() for c in raw.columns]

# 去掉第一行单位说明
raw = raw[pd.to_numeric(raw["T"], errors="coerce").notna()].copy()

# 需要用到的列
use_cols = ["H/C", "GHSV", "T", "fCO2", "fCO", "fH2", "r CH3OH"]

for col in use_cols:
    raw[col] = pd.to_numeric(raw[col], errors="coerce")

# 删除缺失值
df = raw.dropna(subset=use_cols).copy()

# 只保留正值，避免对数和幂函数报错
df = df[
    (df["fCO2"] > 0) &
    (df["fCO"] > 0) &
    (df["fH2"] > 0) &
    (df["r CH3OH"] > 0)
].copy()

print("总有效数据点数 =", len(df))
print("GHSV groups =", sorted(df["GHSV"].unique()))


# =========================
# 2. 定义模型
# =========================
def model_rate(params, T, fCO2, fCO, fH2):
    """
    params = [lnA1, E1, a1, b1, lnA2, E2, a2, b2]
    """
    lnA1, E1, a1, b1, lnA2, E2, a2, b2 = params

    k1 = np.exp(lnA1 - E1 / (R * T))
    k2 = np.exp(lnA2 - E2 / (R * T))

    r1 = k1 * (fCO2 ** a1) * (fH2 ** b1)
    r2 = k2 * (fCO  ** a2) * (fH2 ** b2)

    r_total = r1 + r2
    return r_total, r1, r2


def residuals_log(params, T, fCO2, fCO, fH2, y_obs):
    """
    用 log residual 做拟合，更适合 power law
    """
    y_pred, _, _ = model_rate(params, T, fCO2, fCO, fH2)
    eps = 1e-30
    return np.log(np.maximum(y_pred, eps)) - np.log(np.maximum(y_obs, eps))


def calc_metrics(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return r2, rmse, mape


# =========================
# 3. 单个 GHSV 子集拟合函数
# =========================
def fit_one_group(sub_df, ghsv_value):
    T = sub_df["T"].to_numpy(dtype=float)
    fCO2 = sub_df["fCO2"].to_numpy(dtype=float)
    fCO  = sub_df["fCO"].to_numpy(dtype=float)
    fH2  = sub_df["fH2"].to_numpy(dtype=float)
    y    = sub_df["r CH3OH"].to_numpy(dtype=float)

    # 参数边界
    lower_bounds = np.array([
        -40.0,   0.0,   0.0,  0.0,
        -40.0,   0.0,   0.0,  0.0
    ])

    upper_bounds = np.array([
         20.0, 2.0e5,  4.0,  4.0,
         20.0, 2.0e5,  4.0,  4.0
    ])

    # 多组初值
    initial_guesses = [
        [-5,  80000, 1.0, 2.0,  -5,  80000, 1.0, 2.0],
        [-8,  60000, 1.0, 1.0,  -8,  60000, 1.0, 1.0],
        [-10, 90000, 0.5, 2.5,  -10, 90000, 0.5, 2.5],
        [-3,  50000, 1.5, 1.5,  -3,  50000, 1.5, 1.5],
    ]

    best_res = None
    best_cost = np.inf

    for i, x0 in enumerate(initial_guesses, start=1):
        res = least_squares(
            residuals_log,
            x0=np.array(x0, dtype=float),
            bounds=(lower_bounds, upper_bounds),
            args=(T, fCO2, fCO, fH2, y),
            method="trf",
            loss="soft_l1",
            f_scale=0.1,
            max_nfev=20000
        )
        if res.cost < best_cost:
            best_cost = res.cost
            best_res = res

    params = best_res.x
    y_pred, r1_pred, r2_pred = model_rate(params, T, fCO2, fCO, fH2)
    r2, rmse, mape = calc_metrics(y, y_pred)

    # 参数整理
    lnA1, E1, a1, b1, lnA2, E2, a2, b2 = params
    A1 = np.exp(lnA1)
    A2 = np.exp(lnA2)

    summary = {
        "GHSV": ghsv_value,
        "n_points": len(sub_df),
        "lnA1": lnA1,
        "A1": A1,
        "E1": E1,
        "a1": a1,
        "b1": b1,
        "lnA2": lnA2,
        "A2": A2,
        "E2": E2,
        "a2": a2,
        "b2": b2,
        "R2": r2,
        "RMSE": rmse,
        "MAPE_%": mape,
        "cost": best_res.cost,
        "success": best_res.success,
        "message": best_res.message
    }

    pred_df = sub_df.copy()
    pred_df["r_CH3OH_pred"] = y_pred
    pred_df["r1_CO2_route"] = r1_pred
    pred_df["r2_CO_route"] = r2_pred
    pred_df["frac_CO2_route"] = r1_pred / (r1_pred + r2_pred + 1e-30)
    pred_df["frac_CO_route"] = r2_pred / (r1_pred + r2_pred + 1e-30)
    pred_df["relative_error_%"] = (pred_df["r_CH3OH_pred"] - pred_df["r CH3OH"]) / pred_df["r CH3OH"] * 100

    return summary, pred_df


# =========================
# 4. 按 GHSV 分组拟合
# =========================
all_summary = []
all_pred = []

for ghsv in sorted(df["GHSV"].unique()):
    sub_df = df[df["GHSV"] == ghsv].copy()

    print(f"\n开始拟合 GHSV = {ghsv}")
    print(f"数据点数 = {len(sub_df)}")

    summary, pred_df = fit_one_group(sub_df, ghsv)

    all_summary.append(summary)
    all_pred.append(pred_df)

    print(f"R2   = {summary['R2']:.6f}")
    print(f"RMSE = {summary['RMSE']:.6e}")
    print(f"MAPE = {summary['MAPE_%']:.3f}%")

summary_df = pd.DataFrame(all_summary)
pred_all_df = pd.concat(all_pred, ignore_index=True)


# =========================
# 5. 保存结果
# =========================
output_file = "fit_by_GHSV_MeOH_only.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    summary_df.to_excel(writer, sheet_name="summary_by_GHSV", index=False)

    # 每个 GHSV 单独一个 sheet
    for ghsv in sorted(pred_all_df["GHSV"].unique()):
        tmp = pred_all_df[pred_all_df["GHSV"] == ghsv].copy()
        sheet = f"GHSV_{int(ghsv)}" if float(ghsv).is_integer() else f"GHSV_{ghsv}"
        tmp.to_excel(writer, sheet_name=sheet[:31], index=False)

print(f"\n结果已保存到: {output_file}")


# =========================
# 6. 画每个 GHSV 的 parity plot
# =========================
for ghsv in sorted(pred_all_df["GHSV"].unique()):
    tmp = pred_all_df[pred_all_df["GHSV"] == ghsv].copy()

    y_obs = tmp["r CH3OH"].to_numpy()
    y_pred = tmp["r_CH3OH_pred"].to_numpy()

    mn = min(y_obs.min(), y_pred.min())
    mx = max(y_obs.max(), y_pred.max())

    plt.figure(figsize=(5, 5))
    plt.scatter(y_obs, y_pred)
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("Observed r_CH3OH")
    plt.ylabel("Predicted r_CH3OH")
    plt.title(f"Parity Plot, GHSV = {ghsv}")
    plt.tight_layout()
    plt.show()