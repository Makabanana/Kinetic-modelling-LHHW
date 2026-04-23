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
SKIP_SECOND_ROW = True

# Excel 文件名
EXCEL_FILE = "full data.xlsx"

# 输出文件夹
OUT_DIR = Path("ghsv_split_fit_results_3rxn_Tref")
OUT_DIR.mkdir(exist_ok=True)

# 如果你的 rCO2 是“负值表示消耗”，改成 -1.0
# 如果你的 rCO2 是“正值表示消耗”，保持 1.0
RCO2_SIGN_FACTOR = 1.0

# Tref 计算方式
# "harmonic_invT" 更推荐
# "arithmetic_T" 也可试
TREF_MODE = "harmonic_invT"

# 优化参数
DE_SEED = 42
DE_POPSIZE = 20
DE_MAXITER = 1500
DE_TOL = 1e-6
DE_MUTATION = (0.5, 1.0)
DE_RECOMBINATION = 0.7


# =========================
# 1. 读取数据
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
        "rMeOH", "rCO", "rCO2"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"缺少列: {col}")

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required_cols).copy()

    # 统一 rCO2 符号
    df["rCO2"] = df["rCO2"] * RCO2_SIGN_FACTOR

    print(f"总数据点数: {len(df)}")
    return df


# =========================
# 2. 参考温度 Tref
# =========================
def get_Tref(temperature, mode="harmonic_invT"):
    temperature = np.asarray(temperature, dtype=float)

    if mode == "harmonic_invT":
        # 更适合 Arrhenius 形式
        return 1.0 / np.mean(1.0 / temperature)
    elif mode == "arithmetic_T":
        return np.mean(temperature)
    else:
        raise ValueError("TREF_MODE 只能是 'harmonic_invT' 或 'arithmetic_T'")


def calc_k_Tref(lnA_star, E, temperature, Tref):
    """
    k = A* exp[-E/R * (1/T - 1/Tref)]
    其中 lnA_star = lnA - E/(R*Tref)
    """
    return np.exp(lnA_star - (E / R) * (1.0 / temperature - 1.0 / Tref))


# =========================
# 3. 平衡常数
# 反应1: CO2 + 3H2 -> CH3OH + H2O
# 反应2: CO2 + H2  -> CO + H2O
# 反应3: CO + 2H2  -> CH3OH
# K3 = K1 / K2
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

    K_f1 = np.maximum(K_f1, eps)
    K_f2 = np.maximum(K_f2, eps)
    K_f3 = np.maximum(K_f1 / K_f2, eps)

    return K_f1, K_f2, K_f3


# =========================
# 4. beta 诊断
# =========================
def diagnose_beta(df_group, ghsv_label, save_dir=None):
    T = df_group["T"].to_numpy(dtype=float)
    K_f1, K_f2, K_f3 = calculate_equilibrium_constants(T)

    fCO2 = np.maximum(df_group["fCO2"].to_numpy(dtype=float), eps)
    fH2 = np.maximum(df_group["fH2"].to_numpy(dtype=float), eps)
    fCH3OH = np.maximum(df_group["fCH3OH"].to_numpy(dtype=float), eps)
    fH2O = np.maximum(df_group["fH2O"].to_numpy(dtype=float), eps)
    fCO = np.maximum(df_group["fCO"].to_numpy(dtype=float), eps)

    beta1 = (fCH3OH * fH2O) / np.maximum(K_f1 * fCO2 * (fH2 ** 3), eps)
    beta2 = (fCO * fH2O) / np.maximum(K_f2 * fCO2 * fH2, eps)
    beta3 = fCH3OH / np.maximum(K_f3 * fCO * (fH2 ** 2), eps)

    beta_df = df_group.copy()
    beta_df["beta1"] = beta1
    beta_df["beta2"] = beta2
    beta_df["beta3"] = beta3

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
        "beta2_std": beta2.std(ddof=1) if len(beta2) > 1 else 0.0,

        "beta3_min": beta3.min(),
        "beta3_max": beta3.max(),
        "beta3_mean": beta3.mean(),
        "beta3_median": np.median(beta3),
        "beta3_std": beta3.std(ddof=1) if len(beta3) > 1 else 0.0
    }])

    print(f"\nGHSV = {ghsv_label} 的 beta 诊断")
    print(f"beta1: min={beta1.min():.6e}, max={beta1.max():.6e}, mean={beta1.mean():.6e}")
    print(f"beta2: min={beta2.min():.6e}, max={beta2.max():.6e}, mean={beta2.mean():.6e}")
    print(f"beta3: min={beta3.min():.6e}, max={beta3.max():.6e}, mean={beta3.mean():.6e}")

    if save_dir is not None:
        beta_df.to_excel(save_dir / f"GHSV_{ghsv_label}_beta_values.xlsx", index=False)
        beta_summary.to_excel(save_dir / f"GHSV_{ghsv_label}_beta_summary.xlsx", index=False)

    return beta_df, beta_summary


# =========================
# 5. 3RXN simple power law with Tref
#
# 参数顺序:
# lnA1_star, E1, n1_A, n1_B,
# lnA2_star, E2, n2_A, n2_B,
# lnA3_star, E3, n3_A, n3_B
#
# R1: CO2 + 3H2 -> CH3OH + H2O
# R2: CO2 + H2  -> CO + H2O
# R3: CO + 2H2  -> CH3OH
#
# rMeOH = r1 + r3
# rCO   = r2 - r3
# rCO2  = r1 + r2
# =========================
def calc_predictions_simple_3rxn_Tref(par, fuga, temperature, Tref):
    (
        lnA1_star, E1, n1_A, n1_B,
        lnA2_star, E2, n2_A, n2_B,
        lnA3_star, E3, n3_A, n3_B
    ) = par

    fCO2 = np.maximum(fuga[:, 0], eps)
    fH2 = np.maximum(fuga[:, 1], eps)
    fCH3OH = np.maximum(fuga[:, 2], eps)
    fH2O = np.maximum(fuga[:, 3], eps)
    fCO = np.maximum(fuga[:, 4], eps)

    k1 = calc_k_Tref(lnA1_star, E1, temperature, Tref)
    k2 = calc_k_Tref(lnA2_star, E2, temperature, Tref)
    k3 = calc_k_Tref(lnA3_star, E3, temperature, Tref)

    r1 = k1 * (fCO2 ** n1_A) * (fH2 ** n1_B)
    r2 = k2 * (fCO2 ** n2_A) * (fH2 ** n2_B)
    r3 = k3 * (fCO ** n3_A) * (fH2 ** n3_B)

    rMeOH_pred = r1 + r3
    rCO_pred = r2 - r3
    rCO2_pred = r1 + r2

    return r1, r2, r3, rMeOH_pred, rCO_pred, rCO2_pred


# =========================
# 6. 目标函数
# 同时拟合 rMeOH, rCO, rCO2
# =========================
def objective_simple_3rxn_Tref(par, fuga, temperature, Tref, rMeOH_exp, rCO_exp, rCO2_exp):
    try:
        _, _, _, rMeOH_pred, rCO_pred, rCO2_pred = calc_predictions_simple_3rxn_Tref(
            par, fuga, temperature, Tref
        )

        denom_meoh = np.maximum(np.abs(rMeOH_exp), 1e-6)
        denom_co = np.maximum(np.abs(rCO_exp), 1e-6)
        denom_co2 = np.maximum(np.abs(rCO2_exp), 1e-6)

        sse_meoh = np.sum(((rMeOH_pred - rMeOH_exp) / denom_meoh) ** 2)
        sse_co = np.sum(((rCO_pred - rCO_exp) / denom_co) ** 2)
        sse_co2 = np.sum(((rCO2_pred - rCO2_exp) / denom_co2) ** 2)

        total_sse = sse_meoh + sse_co + sse_co2

        if not np.isfinite(total_sse):
            return 1e30

        return float(total_sse)

    except Exception as e:
        print("objective_simple_3rxn_Tref 出错:", e)
        return 1e30


# =========================
# 7. 评价指标
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
# 8. parity plot
# =========================
def make_parity_plot(exp_data, pred_data, xlabel, ylabel, title, save_path):
    plt.figure(figsize=(6, 6))
    plt.scatter(exp_data, pred_data, alpha=0.75)

    min_val = min(np.min(exp_data), np.min(pred_data))
    max_val = max(np.max(exp_data), np.max(pred_data))

    plt.plot([min_val, max_val], [min_val, max_val], "k--")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# =========================
# 9. 单个模型拟合
# =========================
def fit_model_3rxn_Tref(model_name, objective_func, prediction_func, df_group):
    fuga = df_group[["fCO2", "fH2", "fCH3OH", "fH2O", "fCO"]].to_numpy(dtype=float)
    rMeOH = df_group["rMeOH"].to_numpy(dtype=float)
    rCO = df_group["rCO"].to_numpy(dtype=float)
    rCO2 = df_group["rCO2"].to_numpy(dtype=float)
    temperature = df_group["T"].to_numpy(dtype=float)

    Tref = get_Tref(temperature, mode=TREF_MODE)
    print(f"Tref = {Tref:.6f} K")

    bounds = [
        (-30, 30), (0, 150000), (0, 5), (0, 5),   # R1
        (-30, 30), (0, 150000), (0, 5), (0, 5),   # R2
        (-30, 30), (0, 150000), (0, 5), (0, 5)    # R3
    ]

    result = differential_evolution(
        objective_func,
        bounds=bounds,
        args=(fuga, temperature, Tref, rMeOH, rCO, rCO2),
        seed=DE_SEED,
        popsize=DE_POPSIZE,
        maxiter=DE_MAXITER,
        tol=DE_TOL,
        mutation=DE_MUTATION,
        recombination=DE_RECOMBINATION,
        polish=True,
        workers=1,
        updating="immediate"
    )

    par_opt = result.x

    r1_pred, r2_pred, r3_pred, rMeOH_pred, rCO_pred, rCO2_pred = prediction_func(
        par_opt, fuga, temperature, Tref
    )

    (
        lnA1_star, E1, n1_A, n1_B,
        lnA2_star, E2, n2_A, n2_B,
        lnA3_star, E3, n3_A, n3_B
    ) = par_opt

    # 恢复原始 lnA 与 A，便于解释
    lnA1 = lnA1_star + E1 / (R * Tref)
    lnA2 = lnA2_star + E2 / (R * Tref)
    lnA3 = lnA3_star + E3 / (R * Tref)

    A1 = np.exp(lnA1)
    A2 = np.exp(lnA2)
    A3 = np.exp(lnA3)

    A1_star = np.exp(lnA1_star)
    A2_star = np.exp(lnA2_star)
    A3_star = np.exp(lnA3_star)

    r2_meoh = calc_r2(rMeOH, rMeOH_pred)
    r2_co = calc_r2(rCO, rCO_pred)
    r2_co2 = calc_r2(rCO2, rCO2_pred)
    avg_r2 = np.nanmean([r2_meoh, r2_co, r2_co2])

    rmse_meoh = calc_rmse(rMeOH, rMeOH_pred)
    rmse_co = calc_rmse(rCO, rCO_pred)
    rmse_co2 = calc_rmse(rCO2, rCO2_pred)

    mre_meoh = calc_mre(rMeOH, rMeOH_pred)
    mre_co = calc_mre(rCO, rCO_pred)
    mre_co2 = calc_mre(rCO2, rCO2_pred)

    pred_df = pd.DataFrame(index=df_group.index)
    pred_df["r1_pred"] = r1_pred
    pred_df["r2_pred"] = r2_pred
    pred_df["r3_pred"] = r3_pred

    pred_df["rMeOH_pred"] = rMeOH_pred
    pred_df["rCO_pred"] = rCO_pred
    pred_df["rCO2_pred"] = rCO2_pred

    pred_df["rel_error_rMeOH_%"] = np.abs((rMeOH_pred - rMeOH) / np.maximum(np.abs(rMeOH), 1e-12)) * 100
    pred_df["rel_error_rCO_%"] = np.abs((rCO_pred - rCO) / np.maximum(np.abs(rCO), 1e-12)) * 100
    pred_df["rel_error_rCO2_%"] = np.abs((rCO2_pred - rCO2) / np.maximum(np.abs(rCO2), 1e-12)) * 100

    summary = {
        "model": model_name,
        "Tref_K": Tref,
        "optimizer_success": result.success,
        "optimizer_message": str(result.message),
        "objective": result.fun,

        "r2_meoh": r2_meoh,
        "r2_co": r2_co,
        "r2_co2": r2_co2,
        "avg_r2": avg_r2,

        "rmse_meoh": rmse_meoh,
        "rmse_co": rmse_co,
        "rmse_co2": rmse_co2,

        "mre_meoh_%": mre_meoh,
        "mre_co_%": mre_co,
        "mre_co2_%": mre_co2
    }

    params = {
        "model": model_name,
        "Tref_K": Tref,

        "lnA1_star": lnA1_star,
        "A1_star": A1_star,
        "lnA1": lnA1,
        "A1": A1,
        "E1_J_per_mol": E1,
        "E1_kJ_per_mol": E1 / 1000.0,
        "n1_A": n1_A,
        "n1_B": n1_B,

        "lnA2_star": lnA2_star,
        "A2_star": A2_star,
        "lnA2": lnA2,
        "A2": A2,
        "E2_J_per_mol": E2,
        "E2_kJ_per_mol": E2 / 1000.0,
        "n2_A": n2_A,
        "n2_B": n2_B,

        "lnA3_star": lnA3_star,
        "A3_star": A3_star,
        "lnA3": lnA3,
        "A3": A3,
        "E3_J_per_mol": E3,
        "E3_kJ_per_mol": E3 / 1000.0,
        "n3_A": n3_A,
        "n3_B": n3_B
    }

    print(f"\n模型 {model_name} 拟合完成")
    print("success =", result.success)
    print("message =", result.message)
    print("objective =", result.fun)
    print(f"r2_meoh = {r2_meoh:.6f}")
    print(f"r2_co   = {r2_co:.6f}")
    print(f"r2_co2  = {r2_co2:.6f}")
    print(f"avg_r2  = {avg_r2:.6f}")
    print(f"mre_meoh% = {mre_meoh:.4f}")
    print(f"mre_co%   = {mre_co:.4f}")
    print(f"mre_co2%  = {mre_co2:.4f}")

    return summary, params, pred_df


# =========================
# 10. 拟合单个 GHSV 分组
# =========================
def fit_one_ghsv_group(df_group, ghsv_value):
    ghsv_label = int(round(float(ghsv_value)))
    group_dir = OUT_DIR / f"GHSV_{ghsv_label}"
    group_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print(f"开始拟合 GHSV = {ghsv_label}")
    print(f"该组数据点数 = {len(df_group)}")
    print("=" * 60)

    beta_df, beta_summary = diagnose_beta(df_group, ghsv_label, save_dir=group_dir)

    model_name = "3rxn_simple_powerlaw_Tref"

    summary, params, pred_df = fit_model_3rxn_Tref(
        model_name=model_name,
        objective_func=objective_simple_3rxn_Tref,
        prediction_func=calc_predictions_simple_3rxn_Tref,
        df_group=df_group
    )

    summary_df = pd.DataFrame([summary])
    params_df = pd.DataFrame([params])

    result_df = df_group.copy()
    for col in pred_df.columns:
        result_df[col] = pred_df[col].values

    summary_df.to_excel(group_dir / f"GHSV_{ghsv_label}_summary.xlsx", index=False)
    params_df.to_excel(group_dir / f"GHSV_{ghsv_label}_parameters.xlsx", index=False)
    result_df.to_excel(group_dir / f"GHSV_{ghsv_label}_predictions.xlsx", index=False)

    make_parity_plot(
        exp_data=result_df["rMeOH"].values,
        pred_data=result_df["rMeOH_pred"].values,
        xlabel="Experimental MeOH",
        ylabel="Predicted MeOH",
        title=f"Parity Plot for MeOH, GHSV = {ghsv_label}",
        save_path=group_dir / f"GHSV_{ghsv_label}_parity_meoh.png"
    )

    make_parity_plot(
        exp_data=result_df["rCO"].values,
        pred_data=result_df["rCO_pred"].values,
        xlabel="Experimental CO",
        ylabel="Predicted CO",
        title=f"Parity Plot for CO, GHSV = {ghsv_label}",
        save_path=group_dir / f"GHSV_{ghsv_label}_parity_co.png"
    )

    make_parity_plot(
        exp_data=result_df["rCO2"].values,
        pred_data=result_df["rCO2_pred"].values,
        xlabel="Experimental CO2",
        ylabel="Predicted CO2",
        title=f"Parity Plot for CO2, GHSV = {ghsv_label}",
        save_path=group_dir / f"GHSV_{ghsv_label}_parity_co2.png"
    )

    summary_df.insert(0, "GHSV", ghsv_label)
    params_df.insert(0, "GHSV", ghsv_label)

    return summary_df, params_df, result_df, beta_summary


# =========================
# 11. 主程序
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

        if len(df_group) < 10:
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