import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# 0. 全局设置
# =========================
R = 8.314
eps = 1e-12

# 如果 Excel 第二行是单位行，设为 True
# 如果不是，改成 False
SKIP_SECOND_ROW = True

# Excel 文件名
EXCEL_FILE = "full data.xlsx"

# 输出文件夹
OUT_DIR = Path("ghsv_product_inhibition_fit_results")
OUT_DIR.mkdir(exist_ok=True)


# =========================
# 1. 读取数据
# 默认读取第一个 sheet
# =========================
def load_data():
    if SKIP_SECOND_ROW:
        df = pd.read_excel(EXCEL_FILE, skiprows=[1])
    else:
        df = pd.read_excel(EXCEL_FILE)

    df.columns = df.columns.str.strip()

    # 统一列名
    df = df.rename(columns={
        "r CH3OH": "rMeOH",
        "r CO": "rCO",
        "r CO2": "rCO2"
    })

    print("实际列名如下：")
    print(df.columns.tolist())

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

    print(f"总数据点数: {len(df)}")
    return df


# =========================
# 2. 平衡常数函数
# 这里沿用你前面代码的 K 定义和 beta 写法
# =========================
def calculate_equilibrium_constants(T):
    K_f1 = np.exp(
        1.6654 + 4553.34 / T - 2.72613 * np.log(T)
        - 1.422914e-2 * T + 0.172060e-4 * T**2
        - 1.106294e-8 * T**3 + 0.319698e-11 * T**4
    ) * (0.101325) ** (-2)

    K_f2 = np.exp(
        -11.4998 - 4649.92 / T + 3.2066 * np.log(T)
        - 0.0107251 * T + 0.697955e-5 * T**2
        - 0.336848e-8 * T**3 + 0.811184e-12 * T**4
    )

    return K_f1, K_f2


# =========================
# 3. beta 诊断
# =========================
def diagnose_beta(df_group, ghsv_label, save_dir=None):
    T = df_group["T"].to_numpy(dtype=float)

    K_f1, K_f2 = calculate_equilibrium_constants(T)

    fCO2 = np.maximum(df_group["fCO2"].to_numpy(dtype=float), eps)
    fH2 = np.maximum(df_group["fH2"].to_numpy(dtype=float), eps)
    fCH3OH = np.maximum(df_group["fCH3OH"].to_numpy(dtype=float), eps)
    fH2O = np.maximum(df_group["fH2O"].to_numpy(dtype=float), eps)
    fCO = np.maximum(df_group["fCO"].to_numpy(dtype=float), eps)

    beta1 = (fCH3OH * fH2O) / np.maximum(K_f1 * fCO2 * (fH2 ** 3), eps)
    beta2 = (fCO * fH2O) / np.maximum(K_f2 * fCO2 * fH2, eps)

    beta_df = df_group.copy()
    beta_df["beta1"] = beta1
    beta_df["beta2"] = beta2

    beta_summary = pd.DataFrame([{
        "GHSV": ghsv_label,
        "n_points": len(df_group),

        "beta1_min": beta1.min(),
        "beta1_max": beta1.max(),
        "beta1_mean": beta1.mean(),
        "beta1_median": np.median(beta1),
        "beta1_std": beta1.std(ddof=1) if len(beta1) > 1 else 0.0,

        "beta2_min": beta2.min(),
        "beta2_max": beta2.max(),
        "beta2_mean": beta2.mean(),
        "beta2_median": np.median(beta2),
        "beta2_std": beta2.std(ddof=1) if len(beta2) > 1 else 0.0
    }])

    print(f"\nGHSV = {ghsv_label} 的 beta 诊断")
    print(f"beta1: min = {beta1.min():.6e}, max = {beta1.max():.6e}, mean = {beta1.mean():.6e}, median = {np.median(beta1):.6e}")
    print(f"beta2: min = {beta2.min():.6e}, max = {beta2.max():.6e}, mean = {beta2.mean():.6e}, median = {np.median(beta2):.6e}")

    if save_dir is not None:
        beta_df.to_excel(save_dir / f"GHSV_{ghsv_label}_beta_values.xlsx", index=False)
        beta_summary.to_excel(save_dir / f"GHSV_{ghsv_label}_beta_summary.xlsx", index=False)

    return beta_df, beta_summary


# =========================
# 4. 新模型
#
# MeOH:
# rMeOH = k1(T) * fH2^aH2 * fCO2^aCO2 * (xi + fH2O)^aH2O * (xi + fCH3OH)^aCH3OH * (1 - beta1)
#
# RWGS:
# rRWGS = k2(T) * fH2^uH2 * fCO2^uCO2 * (xi + fH2O)^uH2O * fCO^uCO * (1 - beta2)
#
# 这里用 k_ref + E 的参数化方式
# k(T) = k_ref * exp[-E/R * (1/T - 1/Tref)]
# =========================
def calc_predictions_product_inhibition(par, fuga, temperature, Tref):
    (
        lnkref1, E1, aH2, aCO2, aH2O, aCH3OH,
        lnkref2, E2, uH2, uCO2, uH2O, uCO,
        ln_xi
    ) = par

    kref1 = np.exp(lnkref1)
    kref2 = np.exp(lnkref2)
    xi = np.exp(ln_xi)

    fCO2 = np.maximum(fuga[:, 0], eps)
    fH2 = np.maximum(fuga[:, 1], eps)
    fCH3OH = np.maximum(fuga[:, 2], eps)
    fH2O = np.maximum(fuga[:, 3], eps)
    fCO = np.maximum(fuga[:, 4], eps)

    k1 = kref1 * np.exp(-E1 / R * (1.0 / temperature - 1.0 / Tref))
    k2 = kref2 * np.exp(-E2 / R * (1.0 / temperature - 1.0 / Tref))

    K_f1, K_f2 = calculate_equilibrium_constants(temperature)
    K_f1 = np.maximum(K_f1, eps)
    K_f2 = np.maximum(K_f2, eps)

    beta1 = (fCH3OH * fH2O) / np.maximum(K_f1 * fCO2 * (fH2 ** 3), eps)
    beta2 = (fCO * fH2O) / np.maximum(K_f2 * fCO2 * fH2, eps)

    rMeOH = (
        k1
        * (fH2 ** aH2)
        * (fCO2 ** aCO2)
        * ((xi + fH2O) ** aH2O)
        * ((xi + fCH3OH) ** aCH3OH)
        * (1.0 - beta1)
    )

    rCO = (
        k2
        * (fH2 ** uH2)
        * (fCO2 ** uCO2)
        * ((xi + fH2O) ** uH2O)
        * (fCO ** uCO)
        * (1.0 - beta2)
    )

    return rMeOH, rCO, beta1, beta2, k1, k2, xi


# =========================
# 5. 目标函数
# 用相对误差平方和
# =========================
def objective_product_inhibition(par, fuga, temperature, Tref, rMeOH_exp, rCO_exp):
    try:
        rMeOH_pred, rCO_pred, _, _, _, _, _ = calc_predictions_product_inhibition(par, fuga, temperature, Tref)

        if np.any(~np.isfinite(rMeOH_pred)) or np.any(~np.isfinite(rCO_pred)):
            return 1e30

        denom_meoh = np.maximum(np.abs(rMeOH_exp), 1e-6)
        denom_co = np.maximum(np.abs(rCO_exp), 1e-6)

        sse_meoh = np.sum(((rMeOH_pred - rMeOH_exp) / denom_meoh) ** 2)
        sse_co = np.sum(((rCO_pred - rCO_exp) / denom_co) ** 2)
        total_sse = sse_meoh + sse_co

        if not np.isfinite(total_sse):
            return 1e30

        return float(total_sse)

    except Exception as e:
        print("objective_product_inhibition 出错:", e)
        return 1e30


# =========================
# 6. 评价指标
# =========================
def calc_r2(y_exp, y_pred):
    ss_res = np.sum((y_exp - y_pred) ** 2)
    ss_tot = np.sum((y_exp - np.mean(y_exp)) ** 2)
    if ss_tot < 1e-12:
        return np.nan
    return 1 - ss_res / ss_tot


def calc_rmse(y_exp, y_pred):
    return np.sqrt(np.mean((y_exp - y_pred) ** 2))


def calc_mre(y_exp, y_pred):
    denom = np.maximum(np.abs(y_exp), 1e-12)
    return np.mean(np.abs((y_pred - y_exp) / denom)) * 100


# =========================
# 7. 画图
# =========================
def make_parity_plot(exp_y, pred_y, xlabel, ylabel, title, save_path):
    plt.figure(figsize=(6, 6))
    plt.scatter(exp_y, pred_y, alpha=0.75)

    min_val = min(np.min(exp_y), np.min(pred_y))
    max_val = max(np.max(exp_y), np.max(pred_y))

    plt.plot([min_val, max_val], [min_val, max_val], "k--")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# =========================
# 8. 拟合单个 GHSV 组
# =========================
def fit_model_product_inhibition(df_group):
    fuga = df_group[["fCO2", "fH2", "fCH3OH", "fH2O", "fCO"]].to_numpy(dtype=float)
    rMeOH = df_group["rMeOH"].to_numpy(dtype=float)
    rCO = df_group["rCO"].to_numpy(dtype=float)
    temperature = df_group["T"].to_numpy(dtype=float)

    Tref = float(np.mean(temperature))

    # 参数顺序：
    # lnkref1, E1, aH2, aCO2, aH2O, aCH3OH,
    # lnkref2, E2, uH2, uCO2, uH2O, uCO,
    # ln_xi
    bounds = [
        (-30, 30), (0, 150000), (-2, 5), (-2, 5), (-5, 2), (-5, 2),
        (-30, 30), (0, 150000), (-2, 5), (-2, 5), (-5, 2), (-5, 2),
        (-5, 12)
    ]

    result = differential_evolution(
        objective_product_inhibition,
        bounds=bounds,
        args=(fuga, temperature, Tref, rMeOH, rCO),
        seed=42,
        popsize=18,
        maxiter=700,
        tol=1e-6,
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=True,
        workers=1,
        updating="immediate"
    )

    par_opt = result.x

    rMeOH_pred, rCO_pred, beta1_fit, beta2_fit, k1_fit, k2_fit, xi_fit = calc_predictions_product_inhibition(
        par_opt, fuga, temperature, Tref
    )

    (
        lnkref1, E1, aH2, aCO2, aH2O, aCH3OH,
        lnkref2, E2, uH2, uCO2, uH2O, uCO,
        ln_xi
    ) = par_opt

    kref1 = np.exp(lnkref1)
    kref2 = np.exp(lnkref2)

    r2_meoh = calc_r2(rMeOH, rMeOH_pred)
    r2_co = calc_r2(rCO, rCO_pred)
    avg_r2 = np.nanmean([r2_meoh, r2_co])

    rmse_meoh = calc_rmse(rMeOH, rMeOH_pred)
    rmse_co = calc_rmse(rCO, rCO_pred)

    mre_meoh = calc_mre(rMeOH, rMeOH_pred)
    mre_co = calc_mre(rCO, rCO_pred)

    pred_df = df_group.copy()
    pred_df["rMeOH_pred"] = rMeOH_pred
    pred_df["rCO_pred"] = rCO_pred
    pred_df["beta1_fit"] = beta1_fit
    pred_df["beta2_fit"] = beta2_fit
    pred_df["k1_fit"] = k1_fit
    pred_df["k2_fit"] = k2_fit
    pred_df["rel_error_rMeOH_%"] = np.abs((rMeOH_pred - rMeOH) / np.maximum(np.abs(rMeOH), 1e-12)) * 100
    pred_df["rel_error_rCO_%"] = np.abs((rCO_pred - rCO) / np.maximum(np.abs(rCO), 1e-12)) * 100

    summary = {
        "optimizer_success": result.success,
        "optimizer_message": str(result.message),
        "objective": result.fun,
        "Tref_K": Tref,
        "r2_meoh": r2_meoh,
        "r2_co": r2_co,
        "avg_r2": avg_r2,
        "rmse_meoh": rmse_meoh,
        "rmse_co": rmse_co,
        "mre_meoh_%": mre_meoh,
        "mre_co_%": mre_co
    }

    params = {
        "Tref_K": Tref,

        "lnkref1": lnkref1,
        "kref1": kref1,
        "E1_J_per_mol": E1,
        "E1_kJ_per_mol": E1 / 1000.0,
        "aH2": aH2,
        "aCO2": aCO2,
        "aH2O": aH2O,
        "aCH3OH": aCH3OH,

        "lnkref2": lnkref2,
        "kref2": kref2,
        "E2_J_per_mol": E2,
        "E2_kJ_per_mol": E2 / 1000.0,
        "uH2": uH2,
        "uCO2": uCO2,
        "uH2O": uH2O,
        "uCO": uCO,

        "ln_xi": ln_xi,
        "xi_Pa": xi_fit
    }

    print("\n新模型拟合完成")
    print("success =", result.success)
    print("message =", result.message)
    print("objective =", result.fun)
    print(f"Tref = {Tref:.2f} K")
    print(f"r2_meoh = {r2_meoh:.6f}")
    print(f"r2_co   = {r2_co:.6f}")
    print(f"avg_r2  = {avg_r2:.6f}")
    print(f"mre_meoh% = {mre_meoh:.4f}")
    print(f"mre_co%   = {mre_co:.4f}")
    print(f"xi = {xi_fit:.6f} Pa")

    return summary, params, pred_df


# =========================
# 9. 拟合单个 GHSV 分组
# =========================
def fit_one_ghsv_group(df_group, ghsv_value):
    ghsv_label = int(round(float(ghsv_value)))
    group_dir = OUT_DIR / f"GHSV_{ghsv_label}"
    group_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print(f"开始拟合 GHSV = {ghsv_label}")
    print(f"该组数据点数 = {len(df_group)}")
    print("=" * 60)

    # 先做 beta 诊断
    beta_df, beta_summary = diagnose_beta(df_group, ghsv_label, save_dir=group_dir)

    summary, params, result_df = fit_model_product_inhibition(df_group)

    summary_df = pd.DataFrame([summary])
    params_df = pd.DataFrame([params])

    summary_df.insert(0, "GHSV", ghsv_label)
    params_df.insert(0, "GHSV", ghsv_label)

    summary_df.to_excel(group_dir / f"GHSV_{ghsv_label}_summary.xlsx", index=False)
    params_df.to_excel(group_dir / f"GHSV_{ghsv_label}_parameters.xlsx", index=False)
    result_df.to_excel(group_dir / f"GHSV_{ghsv_label}_predictions.xlsx", index=False)

    make_parity_plot(
        exp_y=result_df["rMeOH"].values,
        pred_y=result_df["rMeOH_pred"].values,
        xlabel="Experimental MeOH",
        ylabel="Predicted MeOH",
        title=f"Parity Plot for MeOH, GHSV = {ghsv_label}",
        save_path=group_dir / f"GHSV_{ghsv_label}_parity_meoh.png"
    )

    make_parity_plot(
        exp_y=result_df["rCO"].values,
        pred_y=result_df["rCO_pred"].values,
        xlabel="Experimental CO",
        ylabel="Predicted CO",
        title=f"Parity Plot for CO, GHSV = {ghsv_label}",
        save_path=group_dir / f"GHSV_{ghsv_label}_parity_co.png"
    )

    return summary_df, params_df, result_df, beta_summary


# =========================
# 10. 主程序
# =========================
def main():
    df = load_data()

    ghsv_values = sorted(df["GHSV"].dropna().unique())
    print("\n检测到的 GHSV 分组:", ghsv_values)

    all_summary = []
    all_params = []
    all_predictions = []
    all_beta_summary = []

    for ghsv in ghsv_values:
        df_group = df[np.isclose(df["GHSV"], ghsv)].copy()

        if len(df_group) < 8:
            print(f"GHSV = {ghsv} 的数据点太少，跳过")
            continue

        summary_df, params_df, result_df, beta_summary = fit_one_ghsv_group(df_group, ghsv)

        all_summary.append(summary_df)
        all_params.append(params_df)
        all_predictions.append(result_df)
        all_beta_summary.append(beta_summary)

    if len(all_summary) == 0:
        print("没有成功拟合任何 GHSV 组")
        return

    all_summary_df = pd.concat(all_summary, ignore_index=True)
    all_params_df = pd.concat(all_params, ignore_index=True)
    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    all_beta_summary_df = pd.concat(all_beta_summary, ignore_index=True)

    all_summary_df.to_excel(OUT_DIR / "all_ghsv_summary.xlsx", index=False)
    all_params_df.to_excel(OUT_DIR / "all_ghsv_parameters.xlsx", index=False)
    all_predictions_df.to_excel(OUT_DIR / "all_ghsv_predictions.xlsx", index=False)
    all_beta_summary_df.to_excel(OUT_DIR / "all_ghsv_beta_summary.xlsx", index=False)

    print("\n全部 GHSV 分组拟合完成")
    print("\n总汇总表：")
    print(all_summary_df)

    print("\n总参数表：")
    print(all_params_df)

    print("\n总 beta 汇总表：")
    print(all_beta_summary_df)


if __name__ == "__main__":
    main()